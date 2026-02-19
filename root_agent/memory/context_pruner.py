def prune_context(ctx):
    state = ctx.state

    if "market_raw" in state:
        del state["market_raw"]

    if "finance_raw" in state:
        del state["finance_raw"]
