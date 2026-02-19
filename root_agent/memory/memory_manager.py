# from ..logging.logger import logger
# from ..logging.state_logger import log_state

# class MemoryManager:
#     """
#     Central controller for all state writes.
#     """

#     def __init__(self, session_id: str, state: dict):
#         self.session_id = session_id
#         self.state = state

#     def write(self, key: str, value: str):
#         logger.info(f"STATE WRITE: {key}")
#         self.state[key] = value
#         log_state(self.session_id, self.state)


from .sliding_context import SlidingContext

class MemoryManager:
    def __init__(self):
        self.ctx = SlidingContext()

    def store(self, summary):
        self.ctx.add(summary)

    def get_context(self):
        return self.ctx.get()
