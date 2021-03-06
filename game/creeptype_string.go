// Code generated by "stringer -type=CreepType -trimprefix=Creep"; DO NOT EDIT.

package game

import "strconv"

func _() {
	// An "invalid array index" compiler error signifies that the constant values have changed.
	// Re-run the stringer command to generate them again.
	var x [1]struct{}
	_ = x[CreepNone-0]
	_ = x[CreepCheepy-1]
	_ = x[CreepImp-2]
	_ = x[CreepLion-3]
	_ = x[CreepFairy-4]
	_ = x[CreepMummy-5]
	_ = x[CreepDragon-6]
}

const _CreepType_name = "NoneCheepyImpLionFairyMummyDragon"

var _CreepType_index = [...]uint8{0, 4, 10, 13, 17, 22, 27, 33}

func (i CreepType) String() string {
	if i < 0 || i >= CreepType(len(_CreepType_index)-1) {
		return "CreepType(" + strconv.FormatInt(int64(i), 10) + ")"
	}
	return _CreepType_name[_CreepType_index[i]:_CreepType_index[i+1]]
}
