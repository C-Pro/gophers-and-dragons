package main

import (
	"fmt"
	"math/rand"
	"sync"

	"github.com/quasilyte/gophers-and-dragons/game"
	"github.com/quasilyte/gophers-and-dragons/wasm/sim"
	"github.com/quasilyte/gophers-and-dragons/wasm/simstep"
	// "github.com/traefik/yaegi/stdlib"
)

const (
	numParallel = 4
	numRounds   = 5_000_000

	epsilon      = 0.9999 // exploration/exploitation ratio
	epsilonMin   = 0.2
	epsilonDecay = (epsilon - epsilonMin) / float32(numRounds)
)

var (
	allCreeps = []game.CreepType{
		game.CreepCheepy,
		game.CreepImp,
		game.CreepLion,
		game.CreepFairy,
		game.CreepMummy,
		game.CreepDragon,
		game.CreepNone,
	}
	allCards = []game.CardType{
		game.CardAttack,
		game.CardMagicArrow,
		game.CardRetreat,
		game.CardRest,
		game.CardPowerAttack,
		game.CardFirebolt,
		game.CardStun,
		game.CardHeal,
		game.CardParry,
	}
	simConfig = sim.Config{
		Rounds:   10,
		AvatarHP: 40,
		AvatarMP: 40,
	}
)

func main() {

	theGopher := NewGopherBrain()
	tactic := getTactic(theGopher)

	pool := make(chan struct{}, numParallel)

	wg := sync.WaitGroup{}
	wg.Add(numRounds)

	go func() {
		wg.Wait()
		close(pool)
	}()

	for i := 0; i < numParallel; i++ {
		pool <- struct{}{}
	}

	sim := func() {
		res := sim.Run(&simConfig, tactic)
		s := []game.State{}
		a := []game.CardType{}
		score := 0
		hp := simConfig.AvatarHP
		won := false

		var creep game.CreepType
		for _, step := range res {
			switch v := step.(type) {
			case simstep.UpdateScore:
				score = score + v.Delta
			case simstep.UpdateHP:
				hp = hp + v.Delta
			case simstep.SetCreep:
				creep = v.Type
			case simstep.Victory:
				won = true
			case simstep.UseCard:
				s = append(s, game.State{
					Creep: game.Creep{
						Type: creep,
					},
				})
				a = append(a, v.Type)
			}
		}

		score = score + hp
		for i := range s {
			s[i].Score = score
		}

		theGopher.Learn(s, a, won)
		pool <- struct{}{}
		wg.Done()
	}

	for i := 0; i < numRounds; i++ {
		_, ok := <-pool
		if !ok {
			return
		}
		go sim()
	}

	fmt.Printf("BRAIN DUMP:\n%#v\n", theGopher.probVec)
}

type GopherBrain struct {
	probVec   map[game.CreepType]map[game.CardType]float32
	gamesLost int64
	gamesWon  int64
	maxScore  int
	epsilon   float32
	mux       sync.RWMutex
}

func NewGopherBrain() *GopherBrain {
	gb := GopherBrain{
		probVec: make(map[game.CreepType]map[game.CardType]float32),
		epsilon: epsilon, // exploration/exploitation ratio
	}

	for _, creep := range allCreeps {
		gb.probVec[creep] = make(map[game.CardType]float32)

		for _, card := range allCards {
			gb.probVec[creep][card] = rand.Float32()
		}
	}

	return &gb
}

func (gb *GopherBrain) Think(s game.State) game.CardType {
	if rand.Float32() < gb.epsilon {
		// play a random card from the deck
		cards := []game.CardType{}
		for _, card := range s.Deck {
			if card.Count != 0 {
				cards = append(cards, card.Type)
			}
		}
		return cards[rand.Intn(len(cards))]
	}

	gb.mux.RLock()
	defer gb.mux.RUnlock()

	maxProb := float32(0)
	bestCard := game.CardAttack

	for _, card := range s.Deck {
		if card.Count == 0 {
			continue
		}
		prob := gb.probVec[s.Creep.Type][card.Type]
		if prob > maxProb {
			maxProb = prob
			bestCard = card.Type
		}
	}

	return bestCard
}

func (gb *GopherBrain) Learn(s []game.State, a []game.CardType, won bool) {
	gb.mux.Lock()
	defer gb.mux.Unlock()

	gb.epsilon = gb.epsilon - epsilonDecay

	if len(s) != len(a) {
		panic("episode states and actions length are not equal")
	}

	if won {
		gb.gamesWon++
		if gb.gamesWon%1000 == 0 {
			fmt.Printf("WIN RATIO IS %0.2f (epsilon = %0.2f, max score = %d, lost = %d)\n",
				float32(gb.gamesWon)/float32(gb.gamesLost+gb.gamesWon),
				gb.epsilon,
				gb.maxScore,
				gb.gamesLost,
			)
			gb.gamesLost = 0
			gb.gamesWon = 0
		}
	} else {
		gb.gamesLost++
	}

	l := float32(len(s))
	for i, state := range s {
		card := a[i]
		prob := gb.probVec[state.Creep.Type][card]

		if state.Score > gb.maxScore {
			gb.maxScore = state.Score
		}

		reward := float32(state.Score) / 150.0
		if reward <= 0.0 {
			reward = 1e-6
		}

		if won {
			/*fmt.Printf("%0.5f + %0.5f - %s:%s\n",
				prob,
				((1.0-prob)*reward)/10.0/l,
				state.Creep.Type.String(),
				card.String(),
			)*/
			gb.probVec[state.Creep.Type][card] = prob + ((1.0-prob)*reward)/4.0/l
		} else {
			/*fmt.Printf("%0.5f - %0.5f - %s:%s\n",
				prob,
				prob/10.0/l,
				state.Creep.Type.String(),
				card.String(),
			)*/
			gb.probVec[state.Creep.Type][card] = prob - prob/2.0/l
		}
	}
}

func getTactic(gb *GopherBrain) func(game.State) game.CardType {
	return func(s game.State) game.CardType {
		return gb.Think(s)
	}
}

func ChooseCard(s game.State) game.CardType {
	brain := map[game.CreepType]map[game.CardType]float32{0: {0: 0.75384897, 1: 0.05078125, 2: 0.95889807, 3: 3.3360186e-21, 4: 0.24151509, 5: 0.31152245, 6: 0.9328464, 7: 0.74184895, 8: 0.801055}, 1: {0: 0.015869156, 1: 0.01562497, 2: 0.9862672, 3: 4.074537e-10, 4: 0.99731004, 5: 0.75249016, 6: 0.612854, 7: 0.01752504, 8: 0.8905667}, 2: {0: 0.9097899, 1: 0.9999993, 2: 0.7724614, 3: 0.875, 4: 0.93382466, 5: 0.97660685, 6: 0.031192735, 7: 0.44932643, 8: 0.42235133}, 3: {0: 1, 1: 0.93756056, 2: 0.7519531, 3: 0.5000001, 4: 0.09985055, 5: 0.02365721, 6: 0.4394512, 7: 0.32665655, 8: 0.56228065}, 4: {0: 0.7502136, 1: 0.50000334, 2: 0.9692378, 3: 0.25006008, 4: 0.6097413, 5: 0.37701842, 6: 0.05904001, 7: 0.99981785, 8: 0.5602494}, 5: {0: 0.00024318707, 1: 0.50012195, 2: 0.99999666, 3: 0.00012207404, 4: 0.117670044, 5: 0.039802257, 6: 0.6970816, 7: 0.40839, 8: 0.5473268}, 6: {0: 0.015518071, 1: 0.5122, 2: 0.99993896, 3: 0.006893158, 4: 0.24719195, 5: 0.7994626, 6: 0.98828125, 7: 0.62052774, 8: 0.98828125}}
	maxProb := float32(0)
	bestCard := game.CardAttack
	for _, card := range s.Deck {
		if card.Count == 0 {
			continue
		}
		prob := brain[s.Creep.Type][card.Type]
		if prob > maxProb {
			maxProb = prob
			bestCard = card.Type
		}
	}

	return bestCard
}
