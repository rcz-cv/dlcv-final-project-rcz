import numpy as np

from .database import IdentityDatabase
from .history import IdentityHistory


def tlwh_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    ax2 = ax + aw
    ay2 = ay + ah
    bx2 = bx + bw
    by2 = by + bh

    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    intersection = iw * ih
    union = aw * ah + bw * bh - intersection

    if union <= 0.0:
        return 0.0

    return intersection / union


class IdentityManager:
    def __init__(
        self,
        *,
        knn_k,
        knn_min_votes,
        id_window,
        identity_max_distance,
        min_majority_count,
        conflict_policy,
        max_features_per_identity=100
    ):
        self.database = IdentityDatabase(
            max_distance=identity_max_distance,
            max_features_per_identity=max_features_per_identity,
            knn_k=knn_k,
            knn_min_votes=knn_min_votes,
        )

        self.id_window = id_window
        self.min_majority_count = min_majority_count
        self.conflict_policy = conflict_policy

        self.histories = {}
        self.track_identities = {}
        self.current_frame_idx = None

    def update(self, frame_idx, tracks):
        self.current_frame_idx = frame_idx

        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            feature = getattr(track, "last_feature", None)

            if feature is None:
                continue

            identity_id, distance = self.database.match(feature)

            if identity_id is None:
                identity_id = self.database.create(feature)
            else:
                self.database.add_sample(identity_id, feature)

            self._history(track.track_id).add(frame_idx, identity_id)

            setattr(track, "last_identity_candidate", identity_id)
            setattr(track, "last_identity_distance", distance)

    def resolve_active_track_identities(self, tracks):
        if self.current_frame_idx is None:
            return

        active_track_ids = set()

        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            active_track_ids.add(track.track_id)

            history = self._history(track.track_id)
            identity_id, count = history.majority(self.current_frame_idx)

            if count < self.min_majority_count:
                identity_id = None

            self.track_identities[track.track_id] = identity_id

            setattr(track, "identity_id", identity_id)
            setattr(track, "identity_vote_count", count)

        stale_track_ids = set(self.track_identities.keys()) - active_track_ids
        for track_id in stale_track_ids:
            del self.track_identities[track_id]

    def resolve_conflicts(self, tracks):
        identity_to_tracks = {}

        active_tracks = [
            track for track in tracks
            if track.is_confirmed() and track.time_since_update <= 0
        ]

        for track in active_tracks:
            setattr(track, "identity_conflict", False)

            identity_id = getattr(track, "identity_id", None)
            if identity_id is not None:
                identity_to_tracks.setdefault(identity_id, []).append(track)

        for identity_id, conflicting_tracks in identity_to_tracks.items():
            if len(conflicting_tracks) <= 1:
                continue

            if self.conflict_policy == "mark":
                for track in conflicting_tracks:
                    setattr(track, "identity_conflict", True)

            elif self.conflict_policy == "reset":
                for track in conflicting_tracks:
                    setattr(track, "identity_conflict", True)
                    setattr(track, "identity_id", None)

            elif self.conflict_policy == "competitive":
                winner = max(
                    conflicting_tracks,
                    key=lambda t: (
                        getattr(t, "identity_vote_count", 0),
                        -float(
                            getattr(
                                t,
                                "last_identity_distance",
                                999.0,
                            ) or 999.0
                        ),
                    ),
                )

                for track in conflicting_tracks:
                    if track is winner:
                        continue

                    setattr(track, "identity_conflict", True)
                    setattr(track, "identity_id", None)

            else:
                raise ValueError(
                    f"Unknown conflict policy: {self.conflict_policy}"
                )
 
    def _history(self, track_id):
        if track_id not in self.histories:
            self.histories[track_id] = IdentityHistory(window=self.id_window)
        return self.histories[track_id]
