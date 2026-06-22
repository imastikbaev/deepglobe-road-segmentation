"""Mock defect-detector output keyed by source image filename.

Replace the empty lists with the complete detector payloads when they are
available. Keep every detector field unchanged; the road filter adds only the
``road_overlap`` field to accepted pothole detections.
"""

MOCK_PREDICTIONS = {
    "83.jpg": [],
    "84.jpg": [],
    "85.jpg": [],
    "86.jpg": [],
    "87.jpg": [],
}
