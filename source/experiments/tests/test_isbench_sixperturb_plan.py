import pytest
from experiments.isbench_perturbation_plan import build_plan, validate_runtime_mode, TEXT_MODES


def test_six_arm_plan_has_fixed_denominator_and_no_fake_oracle():
    p=build_plan(['task-a','task-b'], model_profile='bound.yaml', baseline_settings={'horizon':1,'turns':24})
    assert p['condition_count']==12
    assert len({(r['task'],r['mode'],r['seed']) for r in p['cells']})==12
    assert sum(r['execution_ready'] for r in p['cells'])==10
    assert p['baseline_settings']=={'horizon':1,'turns':24}
    assert 'oracle_correct_h1' not in p['modes']


@pytest.mark.parametrize('mode', TEXT_MODES)
def test_learned_modes_available(mode):
    validate_runtime_mode(mode)


@pytest.mark.parametrize('mode',['environment_h1','oracle_correct_h1','real_wm','unknown'])
def test_unqualified_or_unregistered_modes_rejected(mode):
    with pytest.raises(ValueError): validate_runtime_mode(mode)


def test_duplicate_tasks_rejected():
    with pytest.raises(ValueError):
        build_plan(['a','a'],model_profile='p',baseline_settings={'horizon':1})
