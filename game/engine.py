from collections.abc import Callable

from game.rules import (
    cards_that_can_disprove,
    is_correct_accusation,
    next_player_id,
    turn_order_after,
)
from models.actions import MakeAccusation, MakeSuggestion, PassTurn, PlayerAction
from models.card import Card
from models.events import (
    AccusationEvent,
    CardRevealEvent,
    GameOverEvent,
    PublicGameEvent,
    Suggestion,
    SuggestionEvent,
)
from models.game_state import GameState, GameStatus
from models.observation import PlayerObservation
from models.player import LogEntry, Player
from telemetry import tracer
from transport.base import AgentTransport


def default_choose_reveal(player: Player, suggestion: Suggestion, matches: list[Card]) -> Card:
    return matches[0]


class GameEngine:
    """Orchestrates gameplay: validates actions, resolves them into
    authoritative events, updates GameState, and distributes information
    through the transport. Contains no model-reasoning or deduction logic.
    """

    def __init__(
        self,
        state: GameState,
        transport: AgentTransport,
        choose_reveal: Callable[[Player, Suggestion, list[Card]], Card] = default_choose_reveal,
    ) -> None:
        self.state = state
        self.transport = transport
        self.choose_reveal = choose_reveal

    def _player(self, player_id: int) -> Player:
        return next(p for p in self.state.players if p.id == player_id)

    async def take_turn(self, player_id: int, action: PlayerAction) -> PublicGameEvent | None:
        with tracer.start_as_current_span(
            "clue.turn",
            attributes={
                "clue.player_id": player_id,
                "clue.turn_number": self.state.turn_number,
                "clue.action_type": type(action).__name__,
            },
        ):
            if self.state.status == GameStatus.FINISHED:
                raise ValueError("Game is already finished")
            if player_id != self.state.current_player_id:
                raise ValueError(f"It is not player {player_id}'s turn")

            player = self._player(player_id)
            event: PublicGameEvent | None = None

            if isinstance(action, MakeSuggestion):
                if not player.active:
                    raise ValueError(f"Player {player_id} is out and may not suggest")
                suggestion = Suggestion(
                    person=action.person, weapon=action.weapon, room=action.room
                )
                event = await self.resolve_suggestion(player_id, suggestion)
            elif isinstance(action, MakeAccusation):
                if not player.active:
                    raise ValueError(f"Player {player_id} is out and may not accuse")
                accusation = Suggestion(
                    person=action.person, weapon=action.weapon, room=action.room
                )
                event = await self.resolve_accusation(player_id, accusation)
            elif isinstance(action, PassTurn):
                event = None
            else:
                raise TypeError(f"Unknown action type: {type(action)!r}")

            if self.state.status != GameStatus.FINISHED:
                self.state.current_player_id = next_player_id(player_id, self.state.players)
                self.state.turn_number += 1

            return event

    async def resolve_suggestion(
        self, suggesting_player_id: int, suggestion: Suggestion
    ) -> SuggestionEvent:
        with tracer.start_as_current_span(
            "clue.suggestion",
            attributes={
                "clue.suggesting_player_id": suggesting_player_id,
                "clue.turn_number": self.state.turn_number,
            },
        ) as span:
            unable_to_disprove: list[int] = []
            disproving_player_id: int | None = None

            for candidate in turn_order_after(suggesting_player_id, self.state.players):
                matches = cards_that_can_disprove(candidate, suggestion)
                if not matches:
                    unable_to_disprove.append(candidate.id)
                    continue

                revealed = self.choose_reveal(candidate, suggestion, matches)
                disproving_player_id = candidate.id

                reveal_event = CardRevealEvent(
                    turn=self.state.turn_number,
                    revealing_player_id=candidate.id,
                    receiving_player_id=suggesting_player_id,
                    card=revealed,
                )
                self.state.private_reveal_history.append(reveal_event)
                await self.transport.send(candidate.id, reveal_event)
                await self.transport.send(suggesting_player_id, reveal_event)

                self._record_known_card(suggesting_player_id, revealed, candidate.id)
                break

            span.set_attribute("clue.unable_to_disprove_count", len(unable_to_disprove))
            if disproving_player_id is not None:
                span.set_attribute("clue.disproving_player_id", disproving_player_id)

            event = SuggestionEvent(
                turn=self.state.turn_number,
                suggesting_player_id=suggesting_player_id,
                suggestion=suggestion,
                unable_to_disprove=unable_to_disprove,
                disproving_player_id=disproving_player_id,
            )
            self.state.public_event_history.append(event)
            await self.transport.broadcast(event)
            return event

    def _record_known_card(self, player_id: int, card: Card, holder_id: int) -> None:
        player = self._player(player_id)
        entry = LogEntry(seen=True, who_has=holder_id)
        if card.type == "person":
            player.log.people[card.value] = entry
        elif card.type == "weapon":
            player.log.weapons[card.value] = entry
        else:
            player.log.rooms[card.value] = entry

    async def resolve_accusation(
        self, accusing_player_id: int, accusation: Suggestion
    ) -> AccusationEvent:
        with tracer.start_as_current_span(
            "clue.accusation",
            attributes={
                "clue.accusing_player_id": accusing_player_id,
                "clue.turn_number": self.state.turn_number,
            },
        ) as span:
            correct = is_correct_accusation(accusation, self.state.solution)
            span.set_attribute("clue.correct", correct)

            event = AccusationEvent(
                turn=self.state.turn_number,
                accusing_player_id=accusing_player_id,
                accusation=accusation,
                correct=correct,
            )
            self.state.public_event_history.append(event)
            await self.transport.broadcast(event)

            if correct:
                await self._end_game(winning_player_id=accusing_player_id)
            else:
                self._player(accusing_player_id).active = False
                if not any(p.active for p in self.state.players):
                    await self._end_game(winning_player_id=None)

            return event

    async def _end_game(self, winning_player_id: int | None) -> None:
        self.state.status = GameStatus.FINISHED
        self.state.winning_player_id = winning_player_id
        solution = self.state.solution
        game_over = GameOverEvent(
            turn=self.state.turn_number,
            winning_player_id=winning_player_id,
            solution=Suggestion(
                person=solution.person, weapon=solution.weapon, room=solution.room
            ),
        )
        self.state.public_event_history.append(game_over)
        await self.transport.broadcast(game_over)

    def observation_for(self, player_id: int) -> PlayerObservation:
        player = self._player(player_id)
        private_reveals = [
            reveal
            for reveal in self.state.private_reveal_history
            if player_id in (reveal.revealing_player_id, reveal.receiving_player_id)
        ]
        return PlayerObservation(
            player_id=player_id,
            turn_number=self.state.turn_number,
            current_player_id=self.state.current_player_id,
            active=player.active,
            own_cards=list(player.cards),
            card_log=player.log,
            public_history=list(self.state.public_event_history),
            private_reveals=private_reveals,
        )
