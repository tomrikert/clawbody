import numpy as np
from reachy_mini.motion.move import Move

from reachy_mini_openclaw.moves import MovementManager


class ExclusiveMove(Move):
    name = "test_move"
    exclusive_tracking = True

    @property
    def duration(self):
        return 1.0

    def evaluate(self, _t):
        return np.eye(4), np.zeros(2), 0.0


class FakeCamera:
    def __init__(self):
        self.is_head_tracking_enabled = True
        self.changes = []

    def set_head_tracking_enabled(self, enabled):
        self.is_head_tracking_enabled = enabled
        self.changes.append(enabled)

    def get_face_tracking_offsets(self):
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.5)


def test_expressive_move_temporarily_owns_tracking():
    camera = FakeCamera()
    manager = MovementManager(object(), camera_worker=camera)
    manager.queue_move(ExclusiveMove())

    manager._poll_signals(10.0)
    manager._manage_move_queue(10.0)
    assert camera.changes == [False]

    # Secondary sound/face/speech offsets must not corrupt choreography.
    manager.state.face_tracking_offsets = (0.0, 0.0, 0.0, 0.0, 0.0, 0.5)
    _, antennas, yaw = manager._get_secondary_pose()
    assert antennas == (0.0, 0.0)
    assert yaw == 0.0

    manager._manage_move_queue(11.1)
    assert camera.changes == [False, True]
