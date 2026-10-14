"""Single-object visual arbitration, separate from route/stop-sign checks."""


def build_perception_box_prompt(candidate):
    if candidate["kind"] == "stationary_vru":
        return """Classify the object at the yellow marked location; image 2 is its unmarked crop.
It was detected as a pedestrian. Return static ONLY if it is clearly an inanimate
object (e.g. a box or bin), keep for a person, cyclist or animal, uncertain if unclear
or occluded. Being motionless is not evidence of being inanimate. The obstacle
will remain in collision checks. Return ONLY JSON with decision (keep/static/uncertain),
confidence (your certainty from 0 to 1), and reasoning (the visible object's identity).
"""
    evidence = {
        "bridgedrive_only": "BridgeDrive detected this object, but LiDAR had insufficient geometric support.",
        "lidar_only": "LiDAR found a non-ground obstacle surface not explained by current BridgeDrive detections.",
    }[candidate["kind"]]
    return f"""You verify ONE proposed physical obstacle for an autonomous vehicle.
The first image is the scene with the candidate's projected location marked in
yellow. The second image is an unmarked crop of that same location.
The rectangle is a proposal, not evidence that an object exists.
Sensor disagreement: {evidence}
Decide whether to KEEP or DROP this candidate in the obstacle list.
- keep: a real physical object occupies the marked location, including vehicles,
  pedestrians, cones, barriers, walls, or other solid obstacles.
- drop: clear visual evidence shows that the proposal is empty space, a road
  marking, shadow, or another non-obstacle artifact.
- uncertain: occlusion, poor visibility, ambiguous projection or insufficient evidence.
Absence of LiDAR support does NOT prove an object is absent. A point cloud does
NOT prove its semantic interpretation. Do not drop a real roadside object merely
because the current trajectory avoids it. Do not infer absence behind an occluder.
Return ONLY JSON:
{{"decision": "keep|drop|uncertain", "confidence": 0.0, "reasoning": "short visual evidence"}}
"""
