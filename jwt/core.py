"""A deterministic kernel for reading and validating dotted tokens.

A token is three base64url segments joined by dots: the first one carries the
header, the second one the claim set, and the third one the signature.  This
module decodes the first two, keeps the third as an opaque string, and checks
the claims against a validity window, an issuer and an audience.  No signature
is verified and no cryptography is implemented anywhere in here -- the kernel
only looks at the shape of the string and at the values of the claims.

Nothing is read from the clock, the file system or the network either: the
caller passes the current time into :func:`validate`, so every decision the
kernel makes can be replayed from a test.  ::

    token = "header.claims.signature"
    parsed = parse(token)
    parsed.claims["sub"]
    validate(token, now, issuer="issuer-one", audience="aud-one")

Base64url here means the unpadded URL safe alphabet ``A-Z a-z 0-9 - _``.  A
segment never carries ``+``, ``/`` or ``=``, and a segment whose length is one
more than a multiple of four cannot have come out of a byte string at all.

A claim set may carry ``exp`` (the token stops being valid at that second),
``nbf`` (it is not valid before that second) and ``iat``; all three are whole
seconds.  The window is compared with a leeway so that two machines whose
clocks disagree by a few seconds still agree about a token: the kernel accepts
a token exactly while ``nbf - leeway <= now < exp + leeway``.

The errors raised here form two families.  :class:`MalformedTokenError` and
its subclass :class:`SegmentDecodeError` say that the string is not a readable
token; :class:`ClaimError` and its subclasses say that the token is readable
but that its claims do not pass.  Callers catch the family they care about and
tell the individual subclasses apart when they need to.
"""

import json

# -- constants -------------------------------------------------------------

#: How many segments a token is made of.
SEGMENT_COUNT = 3

#: Names of the segments, in the order they appear between the dots.
SEGMENT_NAMES = ("header", "claims", "signature")

#: The unpadded base64url alphabet, in value order.
ALPHABET = ("ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz"
            "0123456789-_")

_VALUES = {character: index for index, character in enumerate(ALPHABET)}

#: The JSON shape every known claim is allowed to take.  A plain type means
#: "a value of that type", a list means "an array whose items all have the
#: shape inside it", and a tuple means "any of these shapes".
CLAIM_TYPES = {
    "iss": str,
    "sub": str,
    "jti": str,
    "aud": (str, [str]),
    "exp": int,
    "nbf": int,
    "iat": int,
}


# -- errors ----------------------------------------------------------------

class TokenError(Exception):
    """Base class of every error raised by this kernel."""


class MalformedTokenError(TokenError, ValueError):
    """The string handed in is not a well formed token at all.

    It derives from :class:`ValueError` as well, so that callers guarding a
    parse with the builtin errors keep catching it.
    """


class SegmentDecodeError(MalformedTokenError):
    """A segment is not base64url, is not UTF-8 text or is not JSON."""


class ClaimError(TokenError, ValueError):
    """The token is readable, but its claims are not acceptable."""


class ClaimTypeError(ClaimError):
    """A claim does not have the JSON shape its name requires."""


class NotYetValidError(ClaimError):
    """The token was not valid yet at the time it was checked."""


class ExpiredTokenError(ClaimError):
    """The token was no longer valid at the time it was checked."""


class IssuerMismatchError(ClaimError):
    """The ``iss`` claim is not the issuer the caller requires."""


class AudienceMismatchError(ClaimError):
    """The ``aud`` claim does not cover the audience the caller requires."""


# -- the parsed token ------------------------------------------------------

class Token:
    """One parsed token: the two documents and the raw segments."""

    __slots__ = ("header", "claims", "signature", "segments")

    def __init__(self, header, claims, signature, segments):
        self.header = header
        self.claims = claims
        self.signature = signature
        self.segments = segments

    def claim(self, name, default=None):
        """Read one claim out of the claim set."""
        return self.claims.get(name, default)

    def __repr__(self):
        return "Token(segments=%d, claims=%d)" % (len(self.segments),
                                                  len(self.claims))


# -- base64url -------------------------------------------------------------

def b64url_encode(raw):
    """Encode bytes as an unpadded base64url segment."""
    raw = bytes(raw)
    text = []
    for start in range(0, len(raw), 3):
        block = raw[start:start + 3]
        word = int.from_bytes(block, "big") << (8 * (3 - len(block)))
        text.extend(ALPHABET[(word >> shift) & 0x3F]
                    for shift in (18, 12, 6, 0))
    if len(raw) % 3 == 1:
        del text[-2:]
    elif len(raw) % 3 == 2:
        del text[-1:]
    return "".join(text)


def b64url_decode(segment):
    """Decode one unpadded base64url segment into bytes.

    A segment of a length that no byte string can produce, a character that is
    not in the alphabet, and leftover bits that are not zero are all refused
    with :class:`SegmentDecodeError`.
    """
    if len(segment) % 4 == 1:
        raise SegmentDecodeError(
            "base64url segment has an impossible length")
    accumulator = 0
    bits = 0
    output = bytearray()
    for character in segment:
        if character not in _VALUES:
            raise SegmentDecodeError(
                "base64url segment contains a character outside the alphabet")
        value = _VALUES[character]
        accumulator = (accumulator << 6) | value
        bits += 6
        if bits >= 8:
            bits -= 8
            output.append((accumulator >> bits) & 0xFF)
            accumulator &= (1 << bits) - 1
    if bits and accumulator & ((1 << bits) - 1):
        raise SegmentDecodeError("base64url segment has non zero trailing bits")
    return bytes(output)


# -- reading a token -------------------------------------------------------

def split_token(token):
    """Split a token into its three segments."""
    if not isinstance(token, str):
        raise MalformedTokenError("token must be a string")
    segments = token.split(".")
    if len(segments) != SEGMENT_COUNT or any(not segment
                                             for segment in segments):
        raise MalformedTokenError("token must have %d segments" % SEGMENT_COUNT)
    return tuple(segments)


def decode_segment(segment):
    """Decode one base64url segment into the JSON object it carries."""
    if not isinstance(segment, str):
        raise SegmentDecodeError("segment must be a string")
    try:
        document = json.loads(b64url_decode(segment).decode("utf-8"))
    except ValueError:
        raise SegmentDecodeError("segment is not a readable JSON document")
    if not isinstance(document, dict):
        raise SegmentDecodeError("segment does not hold a JSON object")
    return document


def parse(token):
    """Decode the header and the claim set of a token.

    The signature segment is carried along untouched: this kernel does not
    verify signatures.
    """
    header_segment, claims_segment, signature = split_token(token)
    header = decode_segment(header_segment)
    claims = decode_segment(claims_segment)
    return Token(header, claims, signature,
                 (header_segment, claims_segment, signature))


# -- claim checks ----------------------------------------------------------

def _matches(value, shape):
    """True when ``value`` has the JSON shape described by ``shape``."""
    if isinstance(shape, tuple):
        return any(_matches(value, item) for item in shape)
    if isinstance(shape, list):
        return (isinstance(value, list)
                and all(_matches(item, shape[0]) for item in value))
    if shape is int:
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, shape)


def check_claim_types(claims, spec=None):
    """Refuse every claim whose value does not have its declared shape."""
    if spec is None:
        spec = CLAIM_TYPES
    for name in sorted(spec):
        if name in claims and not _matches(claims[name], spec[name]):
            raise ClaimTypeError("%s has the wrong claim type" % name)


def check_window(claims, now, leeway=0):
    """Refuse a token whose validity window does not contain ``now``."""
    not_before = claims.get("nbf")
    expires_at = claims.get("exp")
    if not_before is not None and now < not_before - leeway:
        raise NotYetValidError("token is not valid before %d" % not_before)
    if expires_at is not None and now >= expires_at + leeway:
        raise ExpiredTokenError("token expired at %d" % expires_at)


def check_issuer(claims, issuer):
    """Refuse a token that was not issued by ``issuer``."""
    if issuer is None:
        return
    if claims.get("iss") != issuer:
        raise IssuerMismatchError("token was not issued by %s" % issuer)


def check_audience(claims, audience):
    """Refuse a token that does not cover ``audience``."""
    if audience is None:
        return
    declared = claims.get("aud", [])
    if isinstance(declared, str):
        covered = declared == audience
    elif isinstance(declared, list):
        covered = audience in declared
    else:
        covered = False
    if not covered:
        raise AudienceMismatchError("token is not meant for %s" % audience)


def validate(token, now, issuer=None, audience=None, leeway=0, spec=None):
    """Read a token and check its claims, in a fixed order.

    The claim types are checked first, then the validity window, then the
    issuer and finally the audience; the first check that fails decides which
    error the caller sees.  The parsed token comes back when it passes.
    """
    parsed = parse(token)
    claims = parsed.claims
    check_claim_types(claims, spec)
    check_window(claims, now, leeway)
    check_issuer(claims, issuer)
    check_audience(claims, audience)
    return parsed
