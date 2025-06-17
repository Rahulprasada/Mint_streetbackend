import secrets
from django.core.mail import send_mail
from django.conf import settings
from django.utils import timezone
from datetime import timedelta
from .models import User

class TokenManager:
    def generate_token(self, user):
        token = secrets.token_urlsafe(32)
        expiry = timezone.now() + timedelta(hours=1)
        return token, expiry

    def send_confirmation_email(self, user, token):
        confirmation_url = f"{settings.FRONTEND_URL}/confirm-email/{token}"
        send_mail(
            'Confirm Your Email',
            f'Please click this link to confirm your email: {confirmation_url}',
            settings.EMAIL_HOST_USER,
            [user.email],
            fail_silently=False,
        )

    def send_password_reset_email(self, user, token):
        reset_url = f"{settings.FRONTEND_URL}/reset-password/{token}"
        send_mail(
            'Password Reset Request',
            f'Click this link to reset your password: {reset_url}',
            settings.EMAIL_HOST_USER,
            [user.email],
            fail_silently=False,
        )