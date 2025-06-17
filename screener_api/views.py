import json
import os
import logging
import traceback # Import traceback for detailed logging
from .models import User
from .utils import TokenManager
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from . import screener_logic
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView
from django.utils import timezone
import uuid
from django.core.mail import send_mail

logger = logging.getLogger(__name__) # Get logger for this module


class UserStatusView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        return Response({
            'email': request.user.email,
            'is_active': request.user.is_active,
            'username': request.user.username
        })

class RegisterView(APIView):
    def post(self, request):
        try:
            email = request.data.get('email')
            username = request.data.get('username')
            password = request.data.get('password')

            if not all([email, username, password]):
                return Response(
                    {'error': 'Email, username, and password are required'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            if User.objects.filter(email=email).exists():
                return Response(
                    {'error': 'Email already registered'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            if User.objects.filter(username=username).exists():
                return Response(
                    {'error': 'Username already taken'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            user = User(
                email=email,
                username=username,
                is_active=False,
                email_confirmation_token=str(uuid.uuid4()),
                email_confirmation_expiry=timezone.now() + timezone.timedelta(hours=24)
            )
            user.set_password(password)
            user.save()

            confirmation_url = f"http://localhost:3000/confirm?token={user.email_confirmation_token}"
            try:
                send_mail(
                    subject='Confirm Your Email',
                    message=f'Please confirm your email by clicking: {confirmation_url}',
                    from_email='rahulprasadkpm@gmail.com',
                    recipient_list=[email],
                    fail_silently=False,
                )
            except Exception as e:
                logger.error(f"Failed to send email to {email}: {str(e)}")
                # Optionally delete user if email fails
                user.delete()
                return Response(
                    {'error': 'Failed to send confirmation email. Please try again.'},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

            return Response(
                {'message': 'Registration successful. Please check your email to confirm.'},
                status=status.HTTP_201_CREATED
            )
        except Exception as e:
            logger.error(f"Registration error: {str(e)}")
            return Response(
                {'error': 'An error occurred during registration. Please try again.'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    def post(self, request):
        email = request.data.get('email')
        username = request.data.get('username')
        password = request.data.get('password')

        if not all([email, username, password]):
            return Response(
                {'error': 'Email, username, and password are required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        if User.objects.filter(email=email).exists():
            return Response(
                {'error': 'Email already registered'},
                status=status.HTTP_400_BAD_REQUEST
            )

        if User.objects.filter(username=username).exists():
            return Response(
                {'error': 'Username already taken'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Create user with is_active=False
        user = User(
            email=email,
            username=username,
            is_active=False,
            email_confirmation_token=str(uuid.uuid4()),
            email_confirmation_expiry=timezone.now() + timezone.timedelta(hours=24)
        )
        user.set_password(password)  # Hash the password
        user.save()

        # Send confirmation email
        confirmation_url = f"http://localhost:3000/confirm?token={user.email_confirmation_token}"
        send_mail(
            subject='Confirm Your Email',
            message=f'Please confirm your email by clicking: {confirmation_url}',
            from_email='no-reply@screener.com',
            recipient_list=[email],
            fail_silently=False,
        )

        return Response(
            {'message': 'Registration successful. Please check your email to confirm.'},
            status=status.HTTP_201_CREATED
        )
class ConfirmEmailView(APIView):
    def post(self, request):
        token = request.data.get('token')
        logger.info(f"Received confirmation token: {token}")
        if not token:
            logger.warning("No token provided in request")
            return Response({'error': 'Token is required'}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            user = User.objects.get(email_confirmation_token=token)
            if user.email_confirmation_expiry < timezone.now():
                logger.warning(f"Token expired for user: {user.email}")
                return Response({'error': 'Token expired'}, status=status.HTTP_400_BAD_REQUEST)
            user.is_active = True
            user.email_confirmation_token = None
            user.email_confirmation_expiry = None
            user.save()
            logger.info(f"Email confirmed successfully for user: {user.email}")
            return Response({'message': 'Email confirmed successfully'}, status=status.HTTP_200_OK)
        except User.DoesNotExist:
            logger.warning(f"Invalid token: {token}")
            return Response({'error': 'Invalid token'}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            logger.error(f"Error confirming email: {str(e)}")
            return Response({'error': 'An error occurred. Please try again.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class PasswordResetRequestView(APIView):
    def post(self, request):
        email = request.data.get('email')
        try:
            user = User.objects.get(email=email)
            token_manager = TokenManager()
            token, expiry = token_manager.generate_token(user)
            
            user.password_reset_token = token
            user.password_reset_expiry = expiry
            user.save()
            
            token_manager.send_password_reset_email(user, token)
            return Response({'message': 'Password reset email sent'}, status=status.HTTP_200_OK)
        except User.DoesNotExist:
            return Response({'error': 'User not found'}, status=status.HTTP_404_NOT_FOUND)

class PasswordResetConfirmView(APIView):
    def post(self, request):
        token = request.data.get('token')
        password = request.data.get('password')
        try:
            user = User.objects.get(password_reset_token=token)
            if user.password_reset_expiry < timezone.now():
                return Response({'error': 'Token expired'}, status=status.HTTP_400_BAD_REQUEST)
            
            user.set_password(password)
            user.password_reset_token = None
            user.password_reset_expiry = None
            user.save()
            return Response({'message': 'Password reset successful'}, status=status.HTTP_200_OK)
        except User.DoesNotExist:
            return Response({'error': 'Invalid token'}, status=status.HTTP_400_BAD_REQUEST)

class ProtectedView(APIView):
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        return Response({'message': f'Hello, {request.user.email}! This is a protected endpoint.'})


@api_view(['GET'])
def indices_view(request):
    """
    API endpoint to get predefined indices data.
    """
    logger.info("Received request for indices.")
    try:
        indices_data = screener_logic.get_indices_data()
        return Response(indices_data)
    except Exception as e:
        logger.error(f"Error fetching indices: {e}", exc_info=True)
        return Response({'error': 'Failed to fetch indices data.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
def validate_symbols_view(request):
    """
    API endpoint to validate stock symbols.
    Expects POST data: { "symbols": ["AAPL", "MSFT"], "exchange_suffix": ".NS" }
    """
    logger.info("Received request to validate symbols.")
    symbols = request.data.get('symbols')
    exchange_suffix = request.data.get('exchange_suffix', '.NS') # Default suffix

    if not isinstance(symbols, list) or not symbols:
        logger.warning("Validation request: No symbols list provided.")
        return Response({'error': 'A list of symbols is required.'}, status=status.HTTP_400_BAD_REQUEST)

    # Get default period and interval from logic config or settings if needed
    # For validation, any valid period/interval should work
    period = screener_logic.CONFIG.get('period', '1y')
    interval = screener_logic.CONFIG.get('interval', '1d')

    try:
        # Call the validation logic
        valid_symbols, invalid_symbols_dict = screener_logic.validate_symbols(
            symbols, exchange_suffix, period, interval
        )
        return Response({
            'valid_symbols': valid_symbols,
            'invalid_symbols': invalid_symbols_dict
        })
    except Exception as e:
        logger.error(f"Error validating symbols: {e}", exc_info=True)
        return Response({'error': 'Failed to validate symbols.', 'details': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
def screen_stocks_view(request):
    """
    API endpoint to screen a batch of stocks.
    Expects POST data with screening parameters.
    """
    logger.info("Received request to screen stocks.")
    try:
        # Extract parameters from request data, providing defaults
        symbols = request.data.get('symbols')
        period = request.data.get('period', screener_logic.CONFIG.get('period', '2y'))
        interval = request.data.get('interval', screener_logic.CONFIG.get('interval', '1d'))
        feature_sets = request.data.get('feature_sets', [screener_logic.CONFIG.get('features', ['Returns'])]) # Default feature set
        window = request.data.get('window', screener_logic.CONFIG.get('window', 3))
        max_states = request.data.get('max_states', screener_logic.CONFIG.get('max_states', 2))
        train_window = request.data.get('train_window', screener_logic.CONFIG.get('train_window', 252))
        exchange_suffix = request.data.get('exchange_suffix', '.NS')
        max_workers = request.data.get('max_workers', screener_logic.CONFIG.get('max_workers', 4)) # Use config default
        use_rolling_window = request.data.get('use_rolling_window', screener_logic.CONFIG.get('use_rolling_window', False))
        slippage = request.data.get('slippage', screener_logic.CONFIG.get('slippage', 0.001))


        if not isinstance(symbols, list) or not symbols:
            logger.warning("Screening request: No symbols list provided.")
            return Response({'error': 'A list of symbols is required.'}, status=status.HTTP_400_BAD_REQUEST)

        if not isinstance(feature_sets, list) or not feature_sets or not any(feature_sets):
             logger.warning("Screening request: Invalid or empty feature_sets provided.")
             # Fallback to default if provided feature_sets are invalid
             feature_sets = [screener_logic.CONFIG.get('features', ['Returns'])]
             logger.info(f"Using default feature_sets: {feature_sets}")


        # Call the main screening logic function
        screening_results = screener_logic.screen_stocks(
            symbols, period, interval, feature_sets, window,
            max_states, train_window, exchange_suffix, max_workers,
            use_rolling_window, slippage
        )

        # screening_results is already a list of dictionaries,
        # and DataFrames within results are converted to dict(orient='split')
        # which DRF should handle automatically.

        return Response(screening_results)

    except Exception as e:
        logger.error(f"Error during screen_stocks_view: {e}", exc_info=True)
        # Return a generic error or try to include partial results if possible
        return Response({'error': 'An internal server error occurred during screening.', 'details': str(e)},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET', 'POST'])
def config_view(request):
    """
    API endpoint to get or save configuration.
    GET: Returns current config.
    POST: Expects config dictionary in body to save.
    """
    if request.method == 'GET':
        logger.info("Received request to get config.")
        try:
            config = screener_logic.get_config()
            return Response(config)
        except Exception as e:
            logger.error(f"Error getting config: {e}", exc_info=True)
            return Response({'error': 'Failed to get configuration.', 'details': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    elif request.method == 'POST':
        logger.info("Received request to save config.")
        new_config = request.data # Assuming config dictionary is in the request body
        if not isinstance(new_config, dict):
             logger.warning("Save config request: Invalid data format (not a dictionary).")
             return Response({'error': 'Request body must be a JSON dictionary.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            success, message = screener_logic.save_config(new_config)
            if success:
                return Response({'success': True, 'message': message or 'Configuration saved successfully.'})
            else:
                logger.error(f"Error saving config: {message}")
                return Response({'success': False, 'message': message or 'Failed to save configuration.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        except Exception as e:
            logger.error(f"Unexpected error saving config: {e}", exc_info=True)
            return Response({'success': False, 'message': f'An unexpected error occurred: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
def clear_cache_view(request):
    """
    API endpoint to clear the data cache.
    """
    logger.info("Received request to clear data cache.")
    try:
        success, message = screener_logic.clear_data_cache()
        if success:
            return Response({'success': True, 'message': message or 'Data cache cleared successfully.'})
        else:
            logger.error(f"Error clearing cache: {message}")
            return Response({'success': False, 'message': message or 'Failed to clear data cache.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    except Exception as e:
        logger.error(f"Unexpected error clearing cache: {e}", exc_info=True)
        return Response({'success': False, 'message': f'An unexpected error occurred: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def logs_view(request):
    """
    API endpoint to get the latest logs.
    """
    logger.info("Received request for logs.")
    try:
        # Get logs from screener_logic. Max lines can be a query parameter if needed
        log_content = screener_logic.get_logs(tail_lines=500) # Get last 500 lines
        # Return as plain text response
        return Response(log_content, content_type='text/plain')
    except Exception as e:
        logger.error(f"Error fetching logs: {e}", exc_info=True)
        return Response({'error': 'Failed to fetch logs.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
 