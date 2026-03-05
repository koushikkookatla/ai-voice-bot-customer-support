from app.utils.logger import setup_logger

logger = setup_logger(__name__)

# Intent-to-response mapping (domain-constrained, no hallucination)
INTENT_RESPONSES = {
    "order_status": [
        "I can help you track your order. Please provide your order ID and I'll check the status right away.",
        "To look up your order status, I'll need your order number. Could you please share that?",
        "Let me help you with your order status. Please provide your order ID."
    ],
    "cancel_order": [
        "I understand you'd like to cancel your order. Please provide your order ID and I'll process the cancellation.",
        "To cancel your order, I'll need your order number. Orders can be cancelled within 24 hours of placement.",
        "I can help you cancel your order. Please share your order ID so we can proceed."
    ],
    "refund_request": [
        "I'm sorry to hear you need a refund. Please provide your order ID and reason for the refund request.",
        "To process your refund, I'll need your order ID and a brief description of the issue.",
        "Refund requests are typically processed within 5-7 business days. Please provide your order ID to proceed."
    ],
    "subscription_issue": [
        "I can help you with your subscription issue. Could you please describe the problem you're experiencing?",
        "Let me assist with your subscription. Are you having trouble with billing, access, or plan changes?",
        "I understand you're having a subscription issue. Please provide your account email so I can look into it."
    ],
    "payment_problem": [
        "I'm sorry to hear you're experiencing a payment issue. Could you describe the problem?",
        "Payment issues can sometimes occur due to bank authorization. Please try again or contact your bank.",
        "To resolve your payment issue, I'll need to verify your account. Could you provide your registered email?"
    ],
    "account_access": [
        "I can help you regain access to your account. Please provide your registered email address.",
        "To assist with account access, have you tried resetting your password via the forgot password link?",
        "For account access issues, I'll need to verify your identity. Please provide your email address."
    ],
    "product_inquiry": [
        "I'd be happy to answer your product questions. Which product are you interested in?",
        "Great question! Could you specify which product you'd like more information about?",
        "I can provide detailed product information. Which item are you inquiring about?"
    ],
    "shipping_info": [
        "Standard shipping takes 3-5 business days. Express shipping is available for 1-2 day delivery.",
        "We ship to most locations worldwide. Could you provide your order ID for specific tracking information?",
        "For shipping updates, please provide your order ID and I'll check the latest delivery status."
    ],
    "return_request": [
        "I can help you initiate a return. Items can be returned within 30 days of purchase in original condition.",
        "To start your return, please provide your order ID. We'll send you a prepaid return label.",
        "Returns are easy with us! Please share your order ID and I'll guide you through the return process."
    ],
    "general_complaint": [
        "I'm sorry to hear about your experience. I'd like to help resolve this. Could you provide more details?",
        "Thank you for bringing this to our attention. Please describe your concern so I can assist you better.",
        "I apologize for any inconvenience. Your feedback is important to us. Could you explain the issue?"
    ]
}

FALLBACK_RESPONSE = "I'm sorry, I couldn't understand your request. Could you please rephrase or provide more details?"


class ResponseGenerator:
    """Generates contextual customer support responses based on intent."""

    def __init__(self):
        self._response_count = {intent: 0 for intent in INTENT_RESPONSES}
        logger.info("ResponseGenerator initialized")

    def generate(self, intent: str, context: dict = None) -> str:
        """
        Generate response for given intent.

        Args:
            intent: Predicted intent label
            context: Optional context dict for personalization

        Returns:
            Response text string
        """
        if intent not in INTENT_RESPONSES:
            logger.warning(f"Unknown intent: {intent} - using fallback")
            return FALLBACK_RESPONSE

        responses = INTENT_RESPONSES[intent]
        # Round-robin to vary responses
        idx = self._response_count[intent] % len(responses)
        self._response_count[intent] += 1
        response = responses[idx]

        logger.info(f"Generated response for intent '{intent}': {response[:60]}...")
        return response

    def get_all_intents(self) -> list:
        """Return list of all supported intents."""
        return list(INTENT_RESPONSES.keys())
