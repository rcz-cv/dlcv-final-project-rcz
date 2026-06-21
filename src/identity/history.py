from collections import deque, Counter


class IdentityHistory:
    def __init__(self, window=30):
        self.window = window
        self.items = deque()

    def add(self, frame_idx, identity_id):
        self.items.append((frame_idx, identity_id))
        self.prune(frame_idx)

    def prune(self, current_frame_idx):
        while self.items and self.items[0][0] < current_frame_idx - self.window:
            self.items.popleft()

    def majority(self, current_frame_idx):
        self.prune(current_frame_idx)

        identities = [
            identity_id
            for _, identity_id in self.items
            if identity_id is not None
        ]

        if not identities:
            return None, 0

        counts = Counter(identities)
        identity_id, count = counts.most_common(1)[0]
        return identity_id, count
