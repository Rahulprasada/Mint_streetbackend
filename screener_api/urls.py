# backend/screener_api/urls.py
from django.urls import path
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from screener_api.views import (
    RegisterView,
    ConfirmEmailView,
    PasswordResetRequestView,
    PasswordResetConfirmView,
    ProtectedView,
    UserStatusView,
    indices_view,
    logs_view,
    config_view,
    validate_symbols_view,
    screen_stocks_view,
    clear_cache_view,
)

urlpatterns = [
    # Authentication endpoints
    path('api/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('api/register/', RegisterView.as_view(), name='register'),
    path('api/confirm-email/', ConfirmEmailView.as_view(), name='confirm_email'),
    path('api/password-reset/', PasswordResetRequestView.as_view(), name='password_reset_request'),
    path('api/password-reset-confirm/', PasswordResetConfirmView.as_view(), name='password_reset_confirm'),
    path('api/protected/', ProtectedView.as_view(), name='protected'),
    path('api/user-status/', UserStatusView.as_view(), name='user_status'),
    
    # GET endpoints
    path('api/indices/', indices_view, name='indices'),
    path('api/logs/', logs_view, name='logs'),
    path('api/config/', config_view, name='get_config'),  # GET for fetching config

    # POST endpoints
    path('api/validate_symbols/', validate_symbols_view, name='validate_symbols'),
    path('api/screen_stocks/', screen_stocks_view, name='screen_stocks'),
    path('api/config/save/', config_view, name='save_config'),  # POST for saving config
    path('api/clear_cache/', clear_cache_view, name='clear_cache'),
]