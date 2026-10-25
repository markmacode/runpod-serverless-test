import os

# handler.py loads the model when it's imported, so pick the stand-in before any test
# imports it. Forced rather than defaulted, so a HANDLER_MODE left set in the shell
# can't make the tests load the real model.
os.environ["HANDLER_MODE"] = "mock"
