package sim

import (
	"github.com/quasilyte/gophers-and-dragons/game"
	"github.com/quasilyte/gophers-and-dragons/wasm/gamedata"
)

func newCreep(typ game.CreepType) game.Creep {
	stats := gamedata.GetCreepStats(typ)
	return game.Creep{
		Type:       typ,
		HP:         stats.MaxHP,
		CreepStats: stats,
	}
}

func changeDeckCardCount(deck map[game.CardType]game.Card, typ game.CardType, delta int) {
	card := deck[typ]
	card.Count += delta
	deck[typ] = card
}

func calculateHealed(roll, current, max int) int {
	healed := roll
	afterHeal := current + roll
	if afterHeal > max {
		healed -= afterHeal - max
	}
	return healed
}
