class BiMap:
    def __init__(self):
        self.lane_to_id = {}
        self.id_to_lane = {}

    def add(self, lane_id: str, id_: int):
        """Adds a mapping between lane_id and id."""
        self.lane_to_id[lane_id] = id_
        self.id_to_lane[id_] = lane_id

    def get_id(self, lane_id: str) -> int:
        """Gets the id for a given lane_id."""
        return self.lane_to_id.get(lane_id)

    def get_lane(self, id_: int) -> str:
        """Gets the lane_id for a given id."""
        return self.id_to_lane.get(id_)
