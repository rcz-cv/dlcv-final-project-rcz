import numpy as np


def cosine_distance(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    an = np.linalg.norm(a)
    bn = np.linalg.norm(b)

    if an == 0.0 or bn == 0.0:
        return 1.0

    return 1.0 - float(np.dot(a, b) / (an * bn))


class IdentityCluster:
    def __init__(self, identity_id, first_feature):
        self.identity_id = identity_id
        self.features = []
        self.centroid = None
        self.add(first_feature)

    def add(self, feature):
        feature = np.asarray(feature, dtype=np.float32)
        self.features.append(feature)
        self.centroid = np.mean(self.features, axis=0)


class IdentityDatabase:
    def __init__(self, max_distance=0.25, max_features_per_identity=100):
        self.max_distance = max_distance
        self.max_features_per_identity = max_features_per_identity
        self.clusters = {}
        self.next_identity_id = 1

    def match(self, feature):
        if feature is None or len(self.clusters) == 0:
            return None, None

        best_id = None
        best_distance = None

        for identity_id, cluster in self.clusters.items():
            distance = cosine_distance(feature, cluster.centroid)
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_id = identity_id

        if best_distance is None or best_distance > self.max_distance:
            return None, best_distance

        return best_id, best_distance

    def create(self, feature):
        identity_id = self.next_identity_id
        self.next_identity_id += 1
        self.clusters[identity_id] = IdentityCluster(identity_id, feature)
        return identity_id

    def add_sample(self, identity_id, feature):
        cluster = self.clusters[identity_id]
        cluster.add(feature)

        if len(cluster.features) > self.max_features_per_identity:
            cluster.features = cluster.features[-self.max_features_per_identity:]
            cluster.centroid = np.mean(cluster.features, axis=0)
