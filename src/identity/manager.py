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
        id_window,
        identity_max_distance,
        track_detection_iou,
        min_majority_count,
        reset_conflicts,
        max_features_per_identity=100
    ):
        self.database = IdentityDatabase(
            max_distance=identity_max_distance,
            max_features_per_identity=max_features_per_identity,
        )

        self.id_window = id_window
        self.track_detection_iou = track_detection_iou
        self.min_majority_count = min_majority_count
        self.reset_conflicts = reset_conflicts

        self.histories = {}
        self.track_identities = {}
        self.current_frame_idx = None

    def update(self, frame_idx, tracks, detections):
        self.current_frame_idx = frame_idx

        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            detection = self._best_detection_for_track(track, detections)

            if detection is None:
                continue

            feature = getattr(detection["detection"], "feature", None)
            if feature is None:
                print(f"track {track.track_id}: matched detection has no feature")
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

            winner = max(
                conflicting_tracks,
                key=lambda t: (
                    getattr(t, "identity_vote_count", 0),
                    -float(getattr(t, "last_identity_distance", 999.0) or 999.0),
                ),
            )

            for track in conflicting_tracks:
                if track is not winner:
                    setattr(track, "identity_conflict", True)
 
    def _history(self, track_id):
        if track_id not in self.histories:
            self.histories[track_id] = IdentityHistory(window=self.id_window)
        return self.histories[track_id]

    def _best_detection_for_track(self, track, detections):
        if not detections:
            return None

        track_box = track.to_tlwh()
        tx, ty, tw, th = track_box
        tcx = tx + tw / 2.0
        tcy = ty + th / 2.0

        best = None
        best_score = -1.0

        for i, detection in enumerate(detections):
            detection_box = detection.tlwh
            dx, dy, dw, dh = detection_box
            dcx = dx + dw / 2.0
            dcy = dy + dh / 2.0

            iou = tlwh_iou(track_box, detection_box)

            center_distance = np.sqrt((tcx - dcx) ** 2 + (tcy - dcy) ** 2)
            scale = max(th, dh, 1.0)
            normalized_center_distance = center_distance / scale

            center_score = max(0.0, 1.0 - normalized_center_distance)

            # Prefer IoU, but allow center proximity to rescue mismatched box sizes.
            score = max(iou, 0.5 * center_score)

            if score > best_score:
                best_score = score
                best = {
                    "_index": i,
                    "detection": detection,
                    "iou": iou,
                    "center_score": center_score,
                    "score": score,
                }

        if best is None:
            return None

        if best["iou"] < self.track_detection_iou and best["center_score"] < 0.4:
            return None

        return best
