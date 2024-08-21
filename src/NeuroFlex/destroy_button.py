import os
import logging
from typing import Callable, Optional
import hashlib
import time

class DestroyButton:
    def __init__(self, user_id: str, authentication_func: Callable[[str], bool], destruction_func: Callable[[], None]):
        self.user_id = user_id
        self.authentication_func = authentication_func
        self.destruction_func = destruction_func
        self.confirmation_code = None
        self.confirmation_expiry = None
        self.logger = logging.getLogger(__name__)

    def generate_confirmation_code(self) -> str:
        """Generate a unique confirmation code."""
        timestamp = str(time.time()).encode('utf-8')
        return hashlib.sha256(self.user_id.encode('utf-8') + timestamp).hexdigest()[:8]

    def request_destruction(self) -> str:
        """Request destruction and return a confirmation code."""
        if not self.authentication_func(self.user_id):
            self.logger.warning(f"Unauthorized destruction attempt by user {self.user_id}")
            raise PermissionError("User not authorized to initiate destruction.")

        self.confirmation_code = self.generate_confirmation_code()
        self.confirmation_expiry = time.time() + 300  # Code expires in 5 minutes
        self.logger.info(f"Destruction requested by user {self.user_id}. Confirmation code generated.")
        return self.confirmation_code

    def confirm_destruction(self, confirmation_code: str) -> bool:
        """Confirm destruction with the provided code."""
        if not self.confirmation_code or time.time() > self.confirmation_expiry:
            self.logger.warning(f"Invalid or expired confirmation attempt by user {self.user_id}")
            return False

        if confirmation_code != self.confirmation_code:
            self.logger.warning(f"Incorrect confirmation code provided by user {self.user_id}")
            return False

        try:
            self.destruction_func()
            self.logger.critical(f"Destruction initiated by user {self.user_id}")
            return True
        except Exception as e:
            self.logger.error(f"Destruction failed: {str(e)}")
            raise

    def cancel_destruction(self) -> None:
        """Cancel the destruction request."""
        self.confirmation_code = None
        self.confirmation_expiry = None
        self.logger.info(f"Destruction request cancelled by user {self.user_id}")

class HumanOperatedDestroyButton(DestroyButton):
    def __init__(self, user_id, authentication_func, destruction_func):
        super().__init__(user_id, authentication_func, destruction_func)

    def request_human_confirmation(self):
        # Request human confirmation before proceeding with destruction
        user_input = input("Enter confirmation code to destroy (or 'cancel' to abort): ")
        if user_input.lower() == 'cancel':
            self.cancel_destruction()
            print("Destruction cancelled.")
        elif self.confirm_destruction(user_input):
            print("Destruction confirmed and executed.")
        else:
            print("Incorrect confirmation code. Destruction aborted.")

def example_authentication(user_id: str) -> bool:
    """Example authentication function. Replace with actual authentication logic."""
    return user_id == "authorized_user"

def example_destruction() -> None:
    """Example destruction function. Replace with actual destruction logic."""
    print("System destroyed!")

# Integrate the human-operated button into the main script
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    destroy_button = HumanOperatedDestroyButton("authorized_user", example_authentication, example_destruction)

    try:
        confirmation_code = destroy_button.request_destruction()
        print(f"Confirmation code: {confirmation_code}")

        # Request human confirmation
        destroy_button.request_human_confirmation()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
