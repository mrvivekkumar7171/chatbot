"""Authentication helpers for the Streamlit frontend."""
import binascii

import pyotp
import streamlit as st

from config.settings import TOTP_SECRET

try:
    _TOTP = pyotp.TOTP(TOTP_SECRET)
    _TOTP.now()
except (binascii.Error, TypeError, ValueError) as error:
    raise ValueError(
        "TOTP_SECRET must be a valid Base32 secret for an authenticator app."
    ) from error


def require_totp_authentication() -> bool:
    """
    Render the TOTP login form and return whether the current session is trusted.

    The authentication state is scoped to the current Streamlit browser session.
    """
    if st.session_state.get("authenticated", False):
        return True

    st.title("Sign in")
    st.caption("Enter the current code from your authenticator app.")

    with st.form("totp_login"):
        code = st.text_input(
            "Authenticator code",
            max_chars=6,
            placeholder="000000",
            type="password",
        )
        submitted = st.form_submit_button("Sign in", use_container_width=True)

    if submitted:
        if not code.isdigit() or len(code) != 6:
            st.error("Enter the six-digit authenticator code.")
        elif _TOTP.verify(code):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("That code is invalid or has expired. Try the current code.")

    return False


def logout() -> None:
    """Clear the current browser session's authentication state."""
    st.session_state.pop("authenticated", None)
