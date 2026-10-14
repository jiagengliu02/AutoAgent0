"""SparseDrive recovery verification uses future poses without a duplicate origin."""
import numpy as np

from agenticdriving.orchestrator import Orchestrator
from agenticdriving.verifiers.pdms import PDMSVerifier


class SparseDriveOrchestrator(Orchestrator):
    def _recovery_hold_safety_plan(self, trajectory):
        if isinstance(self.verifier, PDMSVerifier):
            # PDMS already connects the real ego at t=0 to the future poses.
            # Prepending an origin would invent a half-second stop.
            return np.asarray(trajectory, dtype=np.float32)
        return super()._recovery_hold_safety_plan(trajectory)
