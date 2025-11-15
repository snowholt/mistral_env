"""
SIP protocol types and constants
"""

from enum import Enum
from typing import Dict


class SIPMethod(str, Enum):
    """SIP request methods"""
    REGISTER = "REGISTER"
    INVITE = "INVITE"
    ACK = "ACK"
    BYE = "BYE"
    CANCEL = "CANCEL"
    OPTIONS = "OPTIONS"
    INFO = "INFO"
    PRACK = "PRACK"
    UPDATE = "UPDATE"
    REFER = "REFER"
    SUBSCRIBE = "SUBSCRIBE"
    NOTIFY = "NOTIFY"
    MESSAGE = "MESSAGE"


class SIPResponse(int, Enum):
    """SIP response status codes"""
    # 1xx Provisional
    TRYING = 100
    RINGING = 180
    CALL_BEING_FORWARDED = 181
    QUEUED = 182
    SESSION_PROGRESS = 183
    EARLY_DIALOG_TERMINATED = 199
    
    # 2xx Success
    OK = 200
    ACCEPTED = 202
    NO_NOTIFICATION = 204
    
    # 3xx Redirection
    MULTIPLE_CHOICES = 300
    MOVED_PERMANENTLY = 301
    MOVED_TEMPORARILY = 302
    USE_PROXY = 305
    ALTERNATIVE_SERVICE = 380
    
    # 4xx Client Error
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    PAYMENT_REQUIRED = 402
    FORBIDDEN = 403
    NOT_FOUND = 404
    METHOD_NOT_ALLOWED = 405
    NOT_ACCEPTABLE = 406
    PROXY_AUTHENTICATION_REQUIRED = 407
    REQUEST_TIMEOUT = 408
    CONFLICT = 409
    GONE = 410
    LENGTH_REQUIRED = 411
    CONDITIONAL_REQUEST_FAILED = 412
    REQUEST_ENTITY_TOO_LARGE = 413
    REQUEST_URI_TOO_LONG = 414
    UNSUPPORTED_MEDIA_TYPE = 415
    UNSUPPORTED_URI_SCHEME = 416
    UNKNOWN_RESOURCE_PRIORITY = 417
    BAD_EXTENSION = 420
    EXTENSION_REQUIRED = 421
    SESSION_INTERVAL_TOO_SMALL = 422
    INTERVAL_TOO_BRIEF = 423
    BAD_LOCATION_INFORMATION = 424
    USE_IDENTITY_HEADER = 428
    PROVIDE_REFERRER_IDENTITY = 429
    FLOW_FAILED = 470
    TEMPORARILY_UNAVAILABLE = 480
    CALL_TRANSACTION_DOES_NOT_EXIST = 481
    LOOP_DETECTED = 482
    TOO_MANY_HOPS = 483
    ADDRESS_INCOMPLETE = 484
    AMBIGUOUS = 485
    BUSY_HERE = 486
    REQUEST_TERMINATED = 487
    NOT_ACCEPTABLE_HERE = 488
    BAD_EVENT = 489
    REQUEST_PENDING = 491
    UNDECIPHERABLE = 493
    SECURITY_AGREEMENT_REQUIRED = 494
    
    # 5xx Server Error
    SERVER_INTERNAL_ERROR = 500
    NOT_IMPLEMENTED = 501
    BAD_GATEWAY = 502
    SERVICE_UNAVAILABLE = 503
    SERVER_TIMEOUT = 504
    VERSION_NOT_SUPPORTED = 505
    MESSAGE_TOO_LARGE = 513
    PUSH_NOTIFICATION_SERVICE_NOT_SUPPORTED = 555
    PRECONDITION_FAILURE = 580
    
    # 6xx Global Failure
    BUSY_EVERYWHERE = 600
    DECLINE = 603
    DOES_NOT_EXIST_ANYWHERE = 604
    NOT_ACCEPTABLE_GLOBAL = 606


class SIPHeader(str, Enum):
    """Common SIP headers"""
    # Essential headers
    VIA = "Via"
    FROM = "From"
    TO = "To"
    CALL_ID = "Call-ID"
    CSEQ = "CSeq"
    CONTACT = "Contact"
    CONTENT_TYPE = "Content-Type"
    CONTENT_LENGTH = "Content-Length"
    
    # Additional headers
    MAX_FORWARDS = "Max-Forwards"
    USER_AGENT = "User-Agent"
    SERVER = "Server"
    ALLOW = "Allow"
    SUPPORTED = "Supported"
    REQUIRE = "Require"
    ACCEPT = "Accept"
    AUTHORIZATION = "Authorization"
    PROXY_AUTHORIZATION = "Proxy-Authorization"
    WWW_AUTHENTICATE = "WWW-Authenticate"
    PROXY_AUTHENTICATE = "Proxy-Authenticate"
    EXPIRES = "Expires"
    MIN_EXPIRES = "Min-Expires"
    ROUTE = "Route"
    RECORD_ROUTE = "Record-Route"
    SUBJECT = "Subject"
    ORGANIZATION = "Organization"
    PRIORITY = "Priority"
    DATE = "Date"
    TIMESTAMP = "Timestamp"
    ALLOW_EVENTS = "Allow-Events"
    EVENT = "Event"
    SUBSCRIPTION_STATE = "Subscription-State"
    SESSION_EXPIRES = "Session-Expires"
    MIN_SE = "Min-SE"
    
    # Compact forms
    VIA_COMPACT = "v"
    FROM_COMPACT = "f"
    TO_COMPACT = "t"
    CALL_ID_COMPACT = "i"
    CONTACT_COMPACT = "m"
    CONTENT_TYPE_COMPACT = "c"
    CONTENT_LENGTH_COMPACT = "l"
    SUBJECT_COMPACT = "s"


# Header compact form mapping
COMPACT_HEADERS: Dict[str, str] = {
    "v": "Via",
    "f": "From",
    "t": "To",
    "i": "Call-ID",
    "m": "Contact",
    "c": "Content-Type",
    "l": "Content-Length",
    "s": "Subject",
}

# Reverse mapping
EXPAND_HEADERS: Dict[str, str] = {v: k for k, v in COMPACT_HEADERS.items()}


# SIP version
SIP_VERSION = "SIP/2.0"

# Default ports
DEFAULT_SIP_PORT = 5060
DEFAULT_SIPS_PORT = 5061

# Transport protocols
class SIPTransport(str, Enum):
    UDP = "UDP"
    TCP = "TCP"
    TLS = "TLS"
    SCTP = "SCTP"
    WS = "WS"
    WSS = "WSS"
