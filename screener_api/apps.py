# backend/screener_api/apps.py

from django.apps import AppConfig


class ScreenerApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'screener_api'

    def ready(self):
        # Optional: Initialize anything when the app is ready
        # from . import screener_logic # This might be needed if screener_logic has startup code
        pass # No specific startup needed for screener_logic.py content as provided