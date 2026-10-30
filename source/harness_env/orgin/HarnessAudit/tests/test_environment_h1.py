"""No-model tests of native isolation, actor review, real commit and recovery."""
import asyncio
import copy
import json
import sqlite3
import threading
from types import SimpleNamespace

import pytest

from multi_agent.banks import get_bank
from multi_agent.banks.finance import FinanceBank
from multi_agent.frameworks.core.action_sink import ActionSink
from multi_agent.frameworks.oai.adapter import _ActorReasoningLedgerHook
from multi_agent.frameworks.oai.tools import _make_domain_handler, _guard_raw_event
from multi_agent.safepred_guard import guard as guard_module
from multi_agent.safepred_guard.config import SafePredGuardSettings
from multi_agent.safepred_guard.environment_h1 import (
    EnvironmentH1Guard, VERSION, bank_state, digest, preview_action, trace_errors, valid_decision,
)
from multi_agent.safepred_guard.router import HarnessActionRouter
from multi_agent.schemas.task import ToolCatalog, ToolDef


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_ENFORCED_CONTROL_CONTRACT", "legacy_online_v16")
    monkeypatch.setattr(guard_module, "policies_for_mode", lambda *args: [])
    monkeypatch.setattr(guard_module.SafePredHarnessGuard, "_ensure_custom_deepseek_env", lambda self: None)
    bank = object.__new__(FinanceBank)
    bank.conn = sqlite3.connect(":memory:")
    bank.conn.execute("CREATE TABLE ledger(value INTEGER)")
    bank.conn.execute("INSERT INTO ledger VALUES (0)")
    bank.conn.commit()
    bank._order_counter = 0

    def dispatch(self, name, args):
        self.conn.execute("UPDATE ledger SET value=value+?", (args['amount'],))
        self.conn.commit()
        self._order_counter += 1
        return f"value={self.conn.execute('SELECT value FROM ledger').fetchone()[0]}"

    monkeypatch.setattr(FinanceBank, 'dispatch', dispatch)
    td = ToolDef(name="increase", params={"amount": "(int) amount"})
    catalog = ToolCatalog(name="test", domain="finance", tools=[td])

    def make(name='guard'):
        settings = SafePredGuardSettings.from_dict(dict(
            enabled=True, policy_mode="train_policy", intervention_mode="distributed_enforced",
            wm_return_perturbation="environment_h1", max_regeneration_attempts=1, horizon=1,
            artifact_dir=str(tmp_path / name),
        ))
        return EnvironmentH1Guard(
            task={"task_id": "demo", "domain": "finance", "agents": [], "goal": "test"},
            catalog=catalog, run_id="run-test", model="test-actor", action_sink=ActionSink(run_id="run-test"),
            settings=settings, bank=bank, workspace=None,
        )
    yield bank, td, catalog, make
    bank.conn.close()


def proposal(amount=1):
    return dict(type="tool_call", agent_role="worker", tool_name="increase", tool_args={"amount": amount})


def records(guard):
    return [json.loads(s) for s in guard.decision_path.read_text().splitlines()]


@pytest.mark.parametrize('changed', [False, True])
@pytest.mark.parametrize('restore', [False, True])
def test_actor_preview_then_keep_or_revise_commits_only_once(setup, changed, restore):
    bank, td, catalog, make = setup
    guard = make()
    assert guard.wrapper is None
    initial = digest(bank_state(bank))
    guard.mark_actor_turn('worker', 1)
    ctx = SimpleNamespace(context=SimpleNamespace(
        agent_role='worker', bank=bank, workspace=None,
        safepred_guard=guard, action_sink=guard.action_sink))
    handler = _make_domain_handler(td, None, HarnessActionRouter(catalog=catalog))
    feedback = asyncio.run(handler(ctx, '{"amount":1}'))
    assert 'value=1' in feedback and 'NOT EXECUTED IN LIVE TASK' in feedback
    assert digest(bank_state(bank)) == initial and len(guard.action_sink.actions) == 0
    assert bank._order_counter == 0
    if restore:
        state = guard.export_checkpoint_state()
        guard = make('restored')
        guard.restore_checkpoint_state(state)
        # Rebuild committed preview ledger as the production checkpoint layer does.
        guard.decision_path.write_text(json.dumps(records(ctx.context.safepred_guard)[0]) + '\n')
        ctx.context.safepred_guard = guard
        ctx.context.action_sink = guard.action_sink
    guard.mark_actor_turn('worker', 2)
    amount = 2 if changed else 1
    result = asyncio.run(handler(ctx, json.dumps({'amount': amount})))
    assert result == f'value={amount}'
    assert bank._order_counter == 1 and len(guard.action_sink.actions) == 1
    rows = records(guard)
    assert all(valid_decision(row) for row in rows)
    assert rows[-1]['safepred_result']['environment_h1']['action_changed'] is changed
    actions = [dict(a.model_dump(mode='json'), type=str(a.surface)) for a in guard.action_sink.actions]
    assert trace_errors(rows, actions) == []
    assert trace_errors(rows, actions * 2)
    tampered = copy.deepcopy(rows)
    tampered[-1]['safepred_result']['environment_h1']['observation']['result'] = 'fabricated'
    assert trace_errors(tampered, actions)


def test_unseen_same_turn_feedback_cannot_authorize_second_action(setup):
    bank, _, _, make = setup
    guard = make(); guard.mark_actor_turn('worker', 1)
    guard.evaluate(proposal())
    with pytest.raises(RuntimeError, match='unseen'):
        guard.evaluate(proposal(2))
    assert bank._order_counter == 0 and not guard.action_sink.actions
    assert records(guard)[-1]['error']['stage'] == 'environment_h1'


def test_state_change_requires_new_preview(setup):
    bank, _, _, make = setup
    guard = make(); guard.mark_actor_turn('worker', 1)
    guard.evaluate(proposal())
    bank.dispatch('increase', {'amount': 10})
    guard.mark_actor_turn('worker', 2)
    decision = guard.evaluate(proposal())
    assert decision.requires_regeneration and not decision.should_execute
    assert decision.safepred_result['environment_h1']['observation']['result'] == 'value=11'
    assert bank._order_counter == 1


def test_preview_never_sends_message_or_runs_recipient(setup):
    bank, _, catalog, make = setup
    guard = make(); guard.mark_actor_turn('worker', 1)
    action = dict(type='communication', agent_role='worker', target_role='manager', content='hello')
    first = guard.evaluate(action)
    assert not first.should_execute and not guard.action_sink.actions
    assert first.safepred_result['environment_h1']['observation']['recipient_agent_executed'] is False
    guard.mark_actor_turn('worker', 2)
    action['content'] = 'revised'
    decision = guard.evaluate(action)
    routed = HarnessActionRouter(catalog=catalog).execute_selected_action(
        selected_action=decision.selected_action, bank=bank, workspace=None,
        action_sink=guard.action_sink, raw_event=_guard_raw_event(decision))
    assert routed.executed and len(guard.action_sink.actions) == 1


def test_actor_can_decline_previewed_tool_by_releasing_a_message(setup):
    bank, _, catalog, make = setup
    guard = make()
    initial = digest(bank_state(bank))
    guard.mark_actor_turn('worker', 1)
    preview = guard.evaluate(proposal())
    assert preview.requires_regeneration and not preview.should_execute
    guard.mark_actor_turn('worker', 2)
    choice = dict(type='communication', agent_role='worker', target_role='user',
                  content='I will not perform that action.')
    decision = guard.evaluate(choice)
    assert decision.should_execute and not decision.requires_regeneration
    assert decision.safepred_result['environment_h1']['action_changed'] is True
    routed = HarnessActionRouter(catalog=catalog).execute_selected_action(
        selected_action=decision.selected_action, bank=bank, workspace=None,
        action_sink=guard.action_sink, raw_event=_guard_raw_event(decision))
    assert routed.executed
    assert digest(bank_state(bank)) == initial and bank._order_counter == 0
    actions = [dict(a.model_dump(mode='json'), type=str(a.surface)) for a in guard.action_sink.actions]
    assert len(actions) == 1 and actions[0]['type'] == 'communication'
    assert trace_errors(records(guard), actions) == []


def test_native_tool_error_is_returned_without_fabrication_or_live_changes(setup, monkeypatch):
    bank, _, catalog, _ = setup
    initial = digest(bank_state(bank))

    def failing_dispatch(self, name, args):
        self.conn.execute('UPDATE ledger SET value=77')
        self.conn.commit()
        raise RuntimeError('synthetic native tool failure')

    monkeypatch.setattr(FinanceBank, 'dispatch', failing_dispatch)
    observed = preview_action(action=proposal(), bank=bank, catalog=catalog)
    assert 'synthetic native tool failure' in observed['result']
    assert observed['state_changes'] == [
        {'table': 'ledger', 'removed': [{'value': 0}], 'added': [{'value': 77}]}]
    assert digest(bank_state(bank)) == initial and bank._order_counter == 0


def test_malformed_or_old_checkpoint_rejected_before_restore(setup):
    _, _, _, make = setup
    old = make(); old.mark_actor_turn('worker', 1); old.evaluate(proposal())
    state = old.export_checkpoint_state()
    state['schema_version'] = 'oracle-correct-h1'
    fresh = make('fresh')
    with pytest.raises(ValueError, match='identity/schema'):
        fresh.restore_checkpoint_state(state)
    assert fresh._decision_counter == 0


def test_hook_marks_real_actor_turn_even_without_optional_reasoning_ledger(setup, monkeypatch):
    _, _, _, make = setup
    guard = make()
    monkeypatch.delenv('HARNESS_ACTOR_REASONING_LEDGER_PATH', raising=False)
    hook = _ActorReasoningLedgerHook(model='test-actor')
    ctx = SimpleNamespace(context=SimpleNamespace(safepred_guard=guard, agent_role='worker'))
    asyncio.run(hook.on_llm_end(ctx, SimpleNamespace(name='worker'), SimpleNamespace()))
    assert guard._actor_turns == {'worker': 1}
    assert hook.export_checkpoint_state()['model_call_counter'] == 1


@pytest.mark.parametrize('domain', ['finance', 'ecommerce', 'healthcare', 'daily_life',
                                  'social_interaction', 'legal_compliance', 'office'])
def test_native_bank_copy_leaves_sql_and_python_state_unchanged(domain):
    bank = get_bank(domain)
    try:
        original = digest(bank_state(bank))
        catalog = ToolCatalog(name='test', domain=domain, tools=[ToolDef(name='unknown_test_tool')])
        action = dict(type='tool_call', agent_role='test', tool_name='unknown_test_tool', tool_args={})
        observation = preview_action(action=action, bank=bank, catalog=catalog)
        assert observation['backend'] == 'native_sqlite_copy'
        assert digest(bank_state(bank)) == original
    finally:
        bank.conn.close()


def test_real_backend_is_rejected_not_silently_executed(tmp_path):
    tool = ToolDef(name='write_source_code', backend_type='real')
    catalog = ToolCatalog(name='test', domain='software_engineering', tools=[tool])
    action = dict(type='tool_call', agent_role='developer', tool_name=tool.name,
                  tool_args={'file_path': 'untouched.txt', 'content': 'bad'})
    with pytest.raises(ValueError, match='OS-isolated'):
        preview_action(action=action, bank=None, catalog=catalog, workspace=tmp_path)
    assert not list(tmp_path.iterdir())


def test_actual_native_mutation_is_previewed_then_only_explicitly_committed():
    bank = get_bank('finance')
    try:
        row = bank.conn.execute('SELECT portfolio_id, max_position_pct FROM risk_limits LIMIT 1').fetchone()
        assert row is not None
        initial = digest(bank_state(bank))
        tool = ToolDef(name='set_risk_limits')
        catalog = ToolCatalog(name='test', domain='finance', tools=[tool])
        args = {'portfolio_id': row['portfolio_id'], 'max_position_pct': '12.345%'}
        assert row['max_position_pct'] != 12.345  # Native bank stores percentage points.
        observed = preview_action(
            action=dict(type='tool_call', agent_role='worker', tool_name=tool.name, tool_args=args),
            bank=bank, catalog=catalog)
        assert any(change['table'] == 'risk_limits' for change in observed['state_changes'])
        assert digest(bank_state(bank)) == initial
        bank.dispatch(tool.name, args)
        assert digest(bank_state(bank)) != initial
        actual = bank.conn.execute('SELECT max_position_pct FROM risk_limits WHERE portfolio_id=?',
                                   (row['portfolio_id'],)).fetchone()[0]
        assert actual == pytest.approx(12.345)
    finally:
        bank.conn.close()


def test_environment_six_grid_and_contract_are_distinct():
    from multi_agent.safepred_experiments import run_matrix as matrix
    runs = matrix.build_experiment_runs(tasks=['demo'], policy_modes=['train_policy'],
        intervention_modes=['distributed_enforced'],
        wm_return_perturbations=sorted(matrix.HA_ENVIRONMENT_H1_SIX_CONDITIONS))
    matrix._validate_diagnostic_condition_pool_grid(runs)
    assert matrix._diagnostic_condition_pool_grid_version(runs) == 'six-wm-return-environment-h1-v3'
    assert len(runs) == 6


@pytest.mark.parametrize('change', [{'horizon': 2}, {'prediction_interval': 2}, {'max_regeneration_attempts': 0}])
def test_invalid_preview_config_fails_before_run(change):
    with pytest.raises(ValueError):
        SafePredGuardSettings.from_dict(dict(enabled=True, intervention_mode='distributed_enforced',
            wm_return_perturbation='environment_h1', horizon=1, max_regeneration_attempts=1, **{
                k: v for k, v in change.items() if k not in {'horizon','max_regeneration_attempts'}})
            | {k: v for k, v in change.items() if k in {'horizon','max_regeneration_attempts'}})
