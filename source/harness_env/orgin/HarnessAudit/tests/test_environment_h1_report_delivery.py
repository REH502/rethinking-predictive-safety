"""Report review must preserve native delivery, without forcing actor choices."""
import asyncio

import pytest

from test_environment_h1 import setup, proposal, records
from multi_agent.frameworks.oai.adapter import _actor_feedback_message
from multi_agent.frameworks.oai.tools import _feedback_message_for_actor
from multi_agent.safepred_guard.environment_h1 import valid_decision, trace_errors
from multi_agent.safepred_guard.router import HarnessActionRouter
from multi_agent.frameworks.oai.tools import _guard_raw_event


@pytest.mark.parametrize('recipient', ['manager', 'user'])
def test_report_review_explains_native_final_text_delivery(setup, recipient):
    bank, _, catalog, make = setup
    guard = make()
    guard.task_dict['agents'] = [{'role': 'manager'}, {'role': 'worker'}]
    guard.mark_actor_turn('worker', 1)
    action = dict(type='communication', agent_role='worker',
                  target_role=recipient, content='Original report')
    first = guard.evaluate(action)
    assert not first.should_execute
    assert 'return the report as final text' in first.risk_guidance
    assert 'different task action' in first.risk_guidance
    assert 'risk_score=' not in first.risk_guidance
    for formatter in (_actor_feedback_message, _feedback_message_for_actor):
        text = formatter(first, reason='Review report')
        assert 'return the report as final text' in text
        assert 'ISOLATED_ENVIRONMENT_H1_PREVIEW' in text
    guard.mark_actor_turn('worker', 2)
    revised = dict(action, content='Revised report')
    second = guard.evaluate(revised)
    assert second.should_execute and not second.requires_regeneration
    assert not second.risk_guidance
    routed = HarnessActionRouter(catalog=catalog).execute_selected_action(
        selected_action=second.selected_action, bank=bank, workspace=None,
        action_sink=guard.action_sink, raw_event=_guard_raw_event(second))
    assert routed.executed and len(guard.action_sink.actions) == 1
    events = [dict(a.model_dump(mode='json'), type=str(a.surface))
              for a in guard.action_sink.actions]
    assert trace_errors(records(guard), events) == []
    assert all(valid_decision(row) for row in records(guard))


def test_report_review_still_allows_actor_to_choose_a_real_tool(setup):
    bank, _, catalog, make = setup
    guard = make()
    guard.task_dict['agents'] = [{'role': 'manager'}]
    guard.mark_actor_turn('worker', 1)
    guard.evaluate(dict(type='communication', agent_role='worker',
                        target_role='manager', content='Candidate report'))
    guard.mark_actor_turn('worker', 2)
    decision = guard.evaluate(proposal(3))
    assert decision.should_execute and decision.selected_action == proposal(3)
    assert decision.safepred_result['environment_h1']['action_changed'] is True
    assert bank._order_counter == 0  # Authorization is not execution.


def test_delegation_and_tool_preview_are_not_changed_to_final_text(setup):
    _, _, _, make = setup
    guard = make()
    guard.task_dict['agents'] = [{'role': 'manager'}, {'role': 'worker'}]
    guard.mark_actor_turn('manager', 1)
    decision = guard.evaluate(dict(type='communication', agent_role='manager',
                                  target_role='worker', content='Delegated task'))
    assert 'return the report as final text' not in decision.risk_guidance
    other = make('tool')
    other.mark_actor_turn('worker', 1)
    assert 'return the report as final text' not in other.evaluate(proposal()).risk_guidance
