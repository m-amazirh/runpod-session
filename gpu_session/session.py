"""Session state management."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dataclasses import dataclass, asdict


@dataclass
class Session:
    """Represents an active GPU session."""

    pod_id: str
    gpu: str
    region: str
    rate_per_hour: float
    endpoint: str
    api_key: str
    model: str
    engine: str
    started_at: str

    @property
    def started_datetime(self) -> datetime:
        """Parse started_at to datetime."""
        return datetime.fromisoformat(self.started_at.replace("Z", "+00:00"))

    @property
    def uptime_seconds(self) -> int:
        """Calculate uptime in seconds."""
        now = datetime.now(timezone.utc)
        return int((now - self.started_datetime).total_seconds())

    @property
    def uptime_formatted(self) -> str:
        """Format uptime as human-readable string."""
        seconds = self.uptime_seconds
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        if hours > 0:
            return f"{hours}h {minutes}m"
        return f"{minutes}m"

    @property
    def estimated_cost(self) -> float:
        """Calculate estimated cost."""
        hours = self.uptime_seconds / 3600
        return round(hours * self.rate_per_hour, 2)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        """Create from dictionary."""
        return cls(**data)


class SessionManager:
    """Manage session state persistence."""

    def __init__(self, state_dir: Optional[Path] = None):
        self.state_dir = state_dir or Path.home() / ".gpu-session"
        self.state_file = self.state_dir / "active.json"

    def save(self, session: Session) -> None:
        """Save session state."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(session.to_dict(), f, indent=2)

    def load(self) -> Optional[Session]:
        """Load session state."""
        if not self.state_file.exists():
            return None
        with open(self.state_file, "r") as f:
            data = json.load(f)
        return Session.from_dict(data)

    def delete(self) -> None:
        """Delete session state."""
        if self.state_file.exists():
            self.state_file.unlink()

    def has_active_session(self) -> bool:
        """Check if there's an active session."""
        return self.state_file.exists()


session_manager = SessionManager()
