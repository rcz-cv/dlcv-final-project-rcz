import numpy as np
from collections import Counter


def normalize_feature(feature):
    feature = np.asarray(feature, dtype=np.float32)
    norm = np.linalg.norm(feature)
    if norm == 0.0:
        return feature
    return feature / norm


def cosine_distance(a, b):
    a = normalize_feature(a)
    b = normalize_feature(b)
    return 1.0 - float(np.dot(a, b))


class IdentityCluster:
    def __init__(self, identity_id, first_feature, max_features=100):
        self.identity_id = identity_id
        self.max_features = max_features
        self.features = []
        self.centroid = None
        self.add(first_feature)

    def add(self, feature):
        feature = normalize_feature(feature)
        self.features.append(feature)

        if len(self.features) > self.max_features:
            self.features = self.features[-self.max_features:]

        self.centroid = normalize_feature(np.mean(self.features, axis=0))


class IdentityDatabase:
    def __init__(
        self,
        max_distance=0.25,
        max_features_per_identity=100,
        knn_k=5,
        knn_min_votes=1,
    ):
        self.max_distance = max_distance
        self.max_features_per_identity = max_features_per_identity
        self.knn_k = knn_k
        self.knn_min_votes = knn_min_votes

        self.clusters = {}
        self.next_identity_id = 1

    def match(self, feature):
        if feature is None or len(self.clusters) == 0:
            return None, None

        query = normalize_feature(feature)
        gallery_features, gallery_identity_ids = self._gallery()

        if len(gallery_features) == 0:
            return None, None

        # Csine distance is 1 - dot, because features normalized
        distances = 1.0 - np.dot(gallery_features, query)

        order = np.argsort(distances)
        k = min(self.knn_k, len(order))
        neighbor_indices = order[:k]

        best_distance = float(distances[neighbor_indices[0]])

        # Radius filter to create new id when even the nearest descriptor is too far away
        if best_distance > self.max_distance:
            return None, best_distance

        neighbor_identity_ids = [
            gallery_identity_ids[i]
            for i in neighbor_indices
            if distances[i] <= self.max_distance
        ]

        if not neighbor_identity_ids:
            return None, best_distance

        counts = Counter(neighbor_identity_ids)
        identity_id, vote_count = counts.most_common(1)[0]

        if vote_count < self.knn_min_votes:
            return None, best_distance

        # Report the best distance to the winning identity only
        winning_distances = [
            float(distances[i])
            for i in neighbor_indices
            if gallery_identity_ids[i] == identity_id
        ]
        winning_distance = min(winning_distances) if winning_distances else best_distance

        return identity_id, winning_distance

    def create(self, feature):
        identity_id = self.next_identity_id
        self.next_identity_id += 1

        self.clusters[identity_id] = IdentityCluster(
            identity_id,
            feature,
            max_features=self.max_features_per_identity,
        )

        return identity_id

    def add_sample(self, identity_id, feature):
        self.clusters[identity_id].add(feature)

    def _gallery(self):
        features = []
        identity_ids = []

        for identity_id, cluster in self.clusters.items():
            for feature in cluster.features:
                features.append(feature)
                identity_ids.append(identity_id)

        if not features:
            return np.empty((0, 0), dtype=np.float32), []

        return np.vstack(features).astype(np.float32), identity_ids
