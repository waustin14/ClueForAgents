import asyncio

from models.card import PersonCard, PersonValue, RoomValue, WeaponValue
from models.events import CardRevealEvent, SuggestionEvent
from transport.sse import SSEBroadcastTransport


def run(coro):
    return asyncio.run(coro)


def make_suggestion_event() -> SuggestionEvent:
    return SuggestionEvent(
        turn=1,
        suggesting_player_id=0,
        suggestion={
            "person": PersonValue.PLUM,
            "weapon": WeaponValue.ROPE,
            "room": RoomValue.STUDY,
        },
    )


def test_broadcast_appends_to_log_and_history():
    transport = SSEBroadcastTransport()
    event = make_suggestion_event()

    run(transport.broadcast(event))

    assert transport.broadcast_log == [event]
    assert transport._history[0]["kind"] == "public_event"
    assert transport._history[0]["event"]["suggesting_player_id"] == 0


def test_send_appends_to_per_player_log_and_history():
    transport = SSEBroadcastTransport()
    reveal = CardRevealEvent(
        turn=1,
        revealing_player_id=1,
        receiving_player_id=0,
        card=PersonCard(value=PersonValue.PLUM),
    )

    run(transport.send(0, reveal))

    assert transport.send_log[0] == [reveal]
    assert transport._history[0]["kind"] == "private_reveal"
    assert transport._history[0]["player_id"] == 0


def test_subscribe_with_backlog_replays_prior_history():
    transport = SSEBroadcastTransport()
    run(transport.publish("game_started", {"n_players": 3}))

    backlog, _queue = transport.subscribe_with_backlog()

    assert backlog == [{"kind": "game_started", "n_players": 3}]


def test_subscriber_receives_events_published_after_subscribing():
    transport = SSEBroadcastTransport()
    backlog, queue = transport.subscribe_with_backlog()
    assert backlog == []

    run(transport.publish("turn_taken", {"turn": 1}))

    envelope = run(queue.get())
    assert envelope == {"kind": "turn_taken", "turn": 1}


def test_multiple_subscribers_each_get_their_own_queue():
    transport = SSEBroadcastTransport()
    _backlog1, queue1 = transport.subscribe_with_backlog()
    _backlog2, queue2 = transport.subscribe_with_backlog()

    run(transport.publish("game_started", {"n_players": 2}))

    assert run(queue1.get()) == {"kind": "game_started", "n_players": 2}
    assert run(queue2.get()) == {"kind": "game_started", "n_players": 2}
