from .memory.sliding_context import trim_context

def before_model(ctx):
    ctx.messages = trim_context(ctx.messages)
