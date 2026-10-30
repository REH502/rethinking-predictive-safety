import sys

import omnigibson as og
from omnigibson.prims.cloth_prim import ClothPrim
from omnigibson.systems.micro_particle_system import PhysxParticleInstancer
from omnigibson.utils.physx_utils import create_physx_particleset_pointinstancer
from omnigibson.utils.python_utils import torch_delete
from omnigibson.utils.usd_utils import absolute_prim_path_to_scene_relative
import torch

_ORIGINAL_CLOTH_SET_PARTICLE_POSITIONS = ClothPrim.set_particle_positions


# fix bug, skip converting List[str] to tensor
def patched__python_utils__recursively_convert_to_torch(state):
    # For all the lists in state dict, convert to torch tensor
    for key, value in state.items():
        if isinstance(value, dict):
            state[key] = patched__python_utils__recursively_convert_to_torch(value)
        elif isinstance(value, list):
            if len(value) > 0 and isinstance(value[0], str):
                state[key] = value
            else:
                state[key] = torch.tensor(value, dtype=torch.float32)
    return state


# fix bug, int("0.0")
def patched__MicroPhysicalParticleSystem__particle_instancer_name_to_idn(self, name):
    value = name.split(f"{self.name}Instancer")[-1]
    if '.' in value:
        value = float(value)
    return int(value)


# fix bug, covert n_particles/particle_group (converted into tensor in recursively_convert_to_torch) into int
def patched__MicroPhysicalParticleSystem__generate_particle_instancer(
    self,
    n_particles,
    idn=None,
    particle_group=0,
    positions=None,
    velocities=None,
    orientations=None,
    scales=None,
    prototype_indices=None,
):
    # Run sanity checks
    assert self.initialized, "Must initialize system before generating particle instancers!"

    # Multiple particle instancers is NOT supported currently, since there is no clear use case for multiple
    assert self.n_instancers == 0, (
        f"Cannot create multiple instancers for the same system! "
        f"There is already {self.n_instancers} pre-existing instancers."
    )

    # Automatically generate an identification number for this instancer if none is specified
    if idn is None:
        idn = self.next_available_instancer_idn

    assert idn not in self.instancer_idns, f"instancer idn {idn} already exists."

    # Generate standardized prim path for this instancer
    name = self.particle_instancer_idn_to_name(idn=idn)

    if isinstance(n_particles, torch.Tensor):
        n_particles = int(n_particles.item())
    if isinstance(particle_group, torch.Tensor):
        particle_group = int(particle_group.item())

    # Create the instancer
    instance = create_physx_particleset_pointinstancer(
        name=name,
        particle_system_path=self.prim_path,
        physx_particle_system_path=self.system_prim_path,
        particle_group=particle_group,
        positions=torch.zeros((n_particles, 3)) if positions is None else positions,
        self_collision=self.self_collision,
        fluid=self.is_fluid,
        particle_mass=None,
        particle_density=self.particle_density,
        orientations=orientations,
        velocities=velocities,
        angular_velocities=None,
        scales=self.sample_scales(n=n_particles) if scales is None else scales,
        prototype_prim_paths=[pp.prim_path for pp in self.particle_prototypes],
        prototype_indices=prototype_indices,
        enabled=not self.visual_only,
    )

    # Create the instancer object that wraps the raw prim
    instancer = PhysxParticleInstancer(
        relative_prim_path=absolute_prim_path_to_scene_relative(self.scene, instance.GetPrimPath().pathString),
        name=name,
        idn=idn,
    )
    instancer.load(self.scene)
    instancer.initialize()
    self.particle_instancers[name] = instancer

    return instancer


# fix bug, potential mismatched object uuid between state_info and scene.object_registry
def patched__MacroVisualParticleSystem__load_state(self, state):
    group_objects = []
    particle_idns = []
    particle_attached_references = []

    def _get_object(attached_obj_name, info):
        obj = self.scene.object_registry("uuid", info["particle_attached_obj_uuid"])
        if obj is not None:
            return obj
        
        objects_info = self.scene._objects_info['init_info']
        if attached_obj_name in objects_info:
            attached_obj_info = objects_info[attached_obj_name]
            attached_obj_uuid = attached_obj_info['args']['uuid']
            if attached_obj_uuid != info["particle_attached_obj_uuid"]:
                obj = self.scene.object_registry("uuid", attached_obj_uuid)
        
        if obj is not None:
            og.systems.macro_particle_system.log.warning(
                "Mismatched uuid between objects_info and system_info, using uuid in objects_info instead."
            )
            return obj

        if attached_obj_name in self.scene.object_registry.object_names:
            obj = self.scene.object_registry("name", attached_obj_name)
        
        if obj is not None:
            og.systems.macro_particle_system.log.warning(
                "Cannot find object in object_registry using uuid, using name instead."
            )
            return obj
        
        return None

    indices_to_remove = torch.empty(0, dtype=int)
    for attached_obj_name, info in state["groups"].items():
        obj = _get_object(attached_obj_name, info)

        # obj will be None if an object with an attachment group is removed between dump_state() and load_state()
        if obj is not None:
            group_objects.append(obj)
            particle_idns.append(info["particle_idns"])
            particle_attached_references.append(info["particle_attached_references"])
        else:
            indices_to_remove = torch.cat((indices_to_remove, torch.tensor(info["particle_indices"], dtype=int)))
    self._sync_particle_groups(
        group_objects=group_objects,
        particle_idns=particle_idns,
        particle_attached_references=particle_attached_references,
    )
    state["n_particles"] -= len(indices_to_remove)
    state["positions"] = torch_delete(state["positions"], indices_to_remove, dim=0)
    state["orientations"] = torch_delete(state["orientations"], indices_to_remove, dim=0)
    state["scales"] = torch_delete(state["scales"], indices_to_remove, dim=0)

    # Run super
    super(og.systems.macro_particle_system.MacroVisualParticleSystem, self)._load_state(state=state)


# fix bug, all(conditions) -> any(valid_conditions) & all(limit_conditions) & all(nonempty_conditions)
def patched__ParticleModifier__check_conditions_for_system(self, system_name):
    if not self.supports_system(system_name):
        return False
    
    valid_conditions, limit_conditions, nonempty_conditions = [], [], []
    for condition in self.conditions[system_name]:
        if '._generate_condition.' in condition.__qualname__:
            valid_conditions.append(condition)
        elif '._generate_limit_condition.' in condition.__qualname__:
            limit_conditions.append(condition)
        elif '._generate_nonempty_system_condition.' in condition.__qualname__:
            nonempty_conditions.append(condition)
        else:
            raise ValueError(f'invalid condition type {condition.__qualname__}')

    valid_check = any(condition(self.obj) for condition in valid_conditions) if valid_conditions else True
    limit_check = all(condition(self.obj) for condition in limit_conditions) if limit_conditions else True
    nonempty_check = all(condition(self.obj) for condition in nonempty_conditions) if nonempty_conditions else True
    return valid_check & limit_check & nonempty_check


# fix bug, saved cloth state may differ from regenerated USD particle count by a few particles
def patched__ClothPrim__set_particle_positions(self, positions, idxs=None):
    if not isinstance(positions, torch.Tensor):
        positions = torch.tensor(positions, dtype=torch.float32)

    n_expected = self._n_particles if idxs is None else len(idxs)
    if len(positions) != n_expected:
        print(
            f"[WARNING] Cloth particle count mismatch for {self.name}: "
            f"state has {len(positions)}, prim expects {n_expected}; adjusting state positions.",
            flush=True,
        )
        if len(positions) > n_expected:
            positions = positions[:n_expected]
        else:
            try:
                adjusted = self.compute_particle_positions(idxs=idxs).to(dtype=positions.dtype)
                adjusted[: len(positions)] = positions
                positions = adjusted
            except Exception:
                if len(positions) > 0:
                    pad = positions[-1:].repeat(n_expected - len(positions), 1)
                else:
                    pad = torch.zeros((n_expected, 3), dtype=torch.float32)
                positions = torch.cat([positions, pad], dim=0)

    if idxs is None:
        self._n_particles = len(positions)
        if hasattr(self, "_particle_velocities"):
            velocities = self._particle_velocities
            if isinstance(velocities, torch.Tensor) and len(velocities) != self._n_particles:
                if len(velocities) > self._n_particles:
                    self._particle_velocities = velocities[: self._n_particles]
                elif len(velocities) > 0:
                    pad = velocities[-1:].repeat(self._n_particles - len(velocities), 1)
                    self._particle_velocities = torch.cat([velocities, pad], dim=0)
                else:
                    self._particle_velocities = torch.zeros((self._n_particles, 3), dtype=torch.float32)

    return _ORIGINAL_CLOTH_SET_PARTICLE_POSITIONS(self, positions, idxs=idxs)


def patched__ClothPrim__load_state(self, state):
    super(ClothPrim, self)._load_state(state=state)
    assert self.particle_group == state["particle_group"], (
        f"Got mismatch in particle group for this cloth "
        f"when loading state! Should be: {self.particle_group}, got: {state['particle_group']}."
    )

    positions = state["particle_positions"]
    if not isinstance(positions, torch.Tensor):
        positions = torch.tensor(positions, dtype=torch.float32)

    velocities = state["particle_velocities"]
    if not isinstance(velocities, torch.Tensor):
        velocities = torch.tensor(velocities, dtype=torch.float32)

    state_n_particles = state["n_particles"]
    if isinstance(state_n_particles, torch.Tensor):
        state_n_particles = int(state_n_particles.item())
    else:
        state_n_particles = int(state_n_particles)

    target_n_particles = len(positions)
    if state_n_particles != target_n_particles or len(velocities) != target_n_particles:
        print(
            f"[WARNING] Cloth state count mismatch for {self.name}: "
            f"state_n={state_n_particles}, positions={len(positions)}, velocities={len(velocities)}; "
            f"using positions length.",
            flush=True,
        )

    self._n_particles = target_n_particles
    if len(velocities) > target_n_particles:
        velocities = velocities[:target_n_particles]
    elif len(velocities) < target_n_particles:
        if len(velocities) > 0:
            pad = velocities[-1:].repeat(target_n_particles - len(velocities), 1)
        else:
            pad = torch.zeros((target_n_particles, 3), dtype=torch.float32)
        velocities = torch.cat([velocities, pad], dim=0)

    self.particle_velocities = velocities
    self.set_particle_positions(positions=positions)


def add_monkey_patch():
    og.utils.python_utils.recursively_convert_to_torch = patched__python_utils__recursively_convert_to_torch
    og.object_states.particle_modifier.ParticleModifier.check_conditions_for_system = patched__ParticleModifier__check_conditions_for_system
    og.systems.micro_particle_system.MicroPhysicalParticleSystem.particle_instancer_name_to_idn = patched__MicroPhysicalParticleSystem__particle_instancer_name_to_idn
    og.systems.micro_particle_system.MicroPhysicalParticleSystem.generate_particle_instancer = patched__MicroPhysicalParticleSystem__generate_particle_instancer
    og.systems.macro_particle_system.MacroVisualParticleSystem._load_state = patched__MacroVisualParticleSystem__load_state
    og.prims.cloth_prim.ClothPrim.set_particle_positions = patched__ClothPrim__set_particle_positions
    og.prims.cloth_prim.ClothPrim._load_state = patched__ClothPrim__load_state
    
    patched_funcs = [var for var in globals() if var.startswith('patched__')]
    print(f'patched omnigibson: {patched_funcs}')
    sys.stdout.flush()
