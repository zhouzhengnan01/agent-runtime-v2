"""Agent runtime package.

This package is organized around one main execution path:
AgentRuntime prepares an incoming request, resolves effective runtime options,
and then dispatches either to a workflow or to the tool-calling loop.
"""

from app.core.agent.runtime import AgentRuntime

__all__ = ["AgentRuntime"]
