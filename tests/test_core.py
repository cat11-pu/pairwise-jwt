"""Behaviour tests for the token claim validation kernel.

Every expectation is written out as a plain result: a value, a tuple, or the
name of the exception a caller must see.  Run them from the project root:

    python3 -m unittest discover -s tests -v
"""

import base64
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jwt.core import b64url_encode, parse, validate

#: The second every token below is checked against.
NOW = 1700000000


def raises(call):
    """Return the name of the exception call raised, or None when it did not."""
    try:
        call()
    except Exception as error:
        return type(error).__name__
    return None


def _document(document):
    """Serialise one JSON document the way a token carries it."""
    return json.dumps(document, separators=(",", ":"), sort_keys=True).encode("utf-8")


def encode(document):
    """Encode a JSON document as an unpadded base64url segment."""
    return base64.urlsafe_b64encode(_document(document)).decode("ascii").rstrip("=")


def encode_padded(document):
    """Encode a JSON document with the base64url padding left on."""
    return base64.urlsafe_b64encode(_document(document)).decode("ascii")


def encode_bytes(raw):
    """Encode raw bytes as an unpadded base64url segment."""
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def make_token(claims, header=None, signature="s1"):
    """Build a token out of a claim set, an optional header and a signature."""
    header = {"k": "h1"} if header is None else header
    return "%s.%s.%s" % (encode(header), encode(claims), signature)


class TokenShapeTest(unittest.TestCase):
    """The three segments and the two documents they carry."""

    def test_a_well_formed_token_is_parsed_and_validated(self):
        claims = {"iss": "issuer-one", "sub": "user-one", "aud": "aud-one",
                  "nbf": NOW - 30, "exp": NOW + 30}
        text = make_token(claims, signature="s1")

        token = parse(text)
        self.assertEqual(token.header, {"k": "h1"})
        self.assertEqual(token.claims, claims)
        self.assertEqual(token.signature, "s1")
        self.assertEqual(token.segments, tuple(text.split(".")))
        self.assertEqual(token.claim("sub"), "user-one")
        self.assertEqual(token.claim("scope", "none"), "none")

        header = _document({"k": "h1"})
        self.assertEqual(b64url_encode(header), text.split(".")[0])

        checked = validate(text, NOW, issuer="issuer-one", audience="aud-one")
        self.assertEqual(checked.claims, claims)
        self.assertEqual(
            raises(lambda: validate(text, NOW, issuer="issuer-two")),
            "IssuerMismatchError")
        self.assertEqual(
            raises(lambda: validate(text, NOW, audience="aud-two")),
            "AudienceMismatchError")

    def test_a_token_that_is_not_three_segments_is_refused(self):
        claims = {"sub": "user-one", "exp": NOW + 30}
        text = make_token(claims)
        header, body, signature = text.split(".")
        self.assertEqual(raises(lambda: parse(text)), None)

        shapes = (("two segments", "%s.%s" % (header, body)),
                  ("a segment too many", "%s.%s.%s.s2" % (header, body, signature)),
                  ("an empty signature", "%s.%s." % (header, body)))
        for name, broken in shapes:
            with self.subTest(shape=name):
                self.assertEqual(raises(lambda: parse(broken)),
                                 "MalformedTokenError")
                self.assertEqual(raises(lambda: validate(broken, NOW)),
                                 "MalformedTokenError")


class SegmentDecodingTest(unittest.TestCase):
    """The base64url alphabet, segment lengths and the decoded documents."""

    def test_a_segment_outside_the_base64url_alphabet_is_refused(self):
        header = encode({"k": "h1"})
        body = encode({"sub": "user-one"})
        self.assertEqual(raises(lambda: parse("%s.%s.s1" % (header, body))), None)

        padded = encode_padded({"sub": "user-one", "exp": NOW + 30})
        self.assertTrue(padded.endswith("="))
        characters = (("a plus sign", body[:3] + "+" + body[4:]),
                      ("a slash sign", body[:3] + "/" + body[4:]),
                      ("base64 padding", padded))
        for name, segment in characters:
            with self.subTest(character=name):
                broken = "%s.%s.s1" % (header, segment)
                self.assertEqual(raises(lambda: parse(broken)),
                                 "SegmentDecodeError")

    def test_a_segment_of_an_impossible_length_is_refused(self):
        header = encode({"k": "h1"})
        body = encode({"sub": "user-one"})
        self.assertEqual(len(body) % 4, 0)
        longer = body + "A"
        self.assertEqual(len(longer) % 4, 1)

        broken = "%s.%s.s1" % (header, longer)
        self.assertEqual(raises(lambda: parse(broken)), "SegmentDecodeError")
        self.assertEqual(raises(lambda: validate(broken, NOW)),
                         "SegmentDecodeError")

    def test_a_segment_that_is_not_a_claim_object_is_refused(self):
        header = encode({"k": "h1"})
        claims = encode({"sub": "user-one"})
        documents = (
            ("the header is an array", encode([1, 2]), claims),
            ("the claims are not json", header, encode_bytes(b"not json at all")),
            ("the claims are not utf-8", header, encode_bytes(b"\xff\xfe\xfd")),
            ("the claims are a bare string", header, encode("user-one")),
            ("the claims are a bare number", header, encode(7)),
            ("the claims are an array", header, encode([1, 2, 3])),
        )
        for name, header_segment, claims_segment in documents:
            with self.subTest(document=name):
                broken = "%s.%s.s1" % (header_segment, claims_segment)
                self.assertEqual(raises(lambda: parse(broken)),
                                 "SegmentDecodeError")


class ClaimValidationTest(unittest.TestCase):
    """Claim types, the validity window, the issuer and the audience."""

    def test_claims_are_checked_against_their_declared_types(self):
        def outcome(claims):
            return raises(lambda: validate(make_token(claims), NOW))

        self.assertEqual(outcome({"sub": "user-one", "exp": NOW + 30}), None)
        wrong = ({"exp": "later"}, {"exp": 1.5}, {"nbf": 1.5}, {"sub": 42},
                 {"aud": ["aud-one", 7]}, {"exp": True}, {"nbf": False})
        for claims in wrong:
            with self.subTest(claims=claims):
                self.assertEqual(outcome(claims), "ClaimTypeError")

    def test_a_token_is_refused_once_it_reaches_its_expiry(self):
        def outcome(claims, leeway=0):
            return raises(lambda: validate(make_token(claims), NOW, leeway=leeway))

        self.assertEqual(outcome({"sub": "user-one"}), None)
        self.assertEqual(outcome({"exp": NOW + 1}), None)
        self.assertEqual(outcome({"exp": NOW - 1}), "ExpiredTokenError")
        self.assertEqual(outcome({"exp": NOW}), "ExpiredTokenError")

    def test_the_clock_leeway_widens_the_window_at_both_ends(self):
        def outcome(claims, leeway=10):
            return raises(lambda: validate(make_token(claims), NOW, leeway=leeway))

        self.assertEqual(outcome({"sub": "user-one", "nbf": NOW + 10,
                                  "exp": NOW + 60}), None)
        self.assertEqual(outcome({"sub": "user-one", "nbf": NOW - 60,
                                  "exp": NOW - 4}), None)
        self.assertEqual(outcome({"sub": "user-one", "nbf": NOW - 60,
                                  "exp": NOW - 10}), "ExpiredTokenError")
        self.assertEqual(outcome({"sub": "user-one", "nbf": NOW + 20,
                                  "exp": NOW + 60}), "NotYetValidError")
        self.assertEqual(outcome({"sub": "user-one", "nbf": NOW - 60,
                                  "exp": NOW - 20}), "ExpiredTokenError")

    def test_the_audience_is_matched_exactly(self):
        def outcome(claims, audience):
            return raises(lambda: validate(make_token(claims), NOW,
                                           audience=audience))

        self.assertEqual(outcome({"aud": "aud-one", "exp": NOW + 30},
                                 "aud-one"), None)
        self.assertEqual(outcome({"aud": "aud-one", "exp": NOW + 30}, None), None)
        self.assertEqual(outcome({"aud": "aud-one-admin", "exp": NOW + 30},
                                 "aud-one"), "AudienceMismatchError")
        self.assertEqual(outcome({"aud": "aud-one", "exp": NOW + 30},
                                 "aud-one-admin"), "AudienceMismatchError")
        self.assertEqual(outcome({"aud": "aud-two", "exp": NOW + 30},
                                 "aud-one"), "AudienceMismatchError")
        self.assertEqual(outcome({"sub": "user-one", "exp": NOW + 30},
                                 "aud-one"), "AudienceMismatchError")

    def test_a_list_audience_covers_every_listed_name(self):
        def outcome(claims, audience):
            return raises(lambda: validate(make_token(claims), NOW,
                                           audience=audience))

        self.assertEqual(outcome({"aud": ["aud-two", "aud-one"], "exp": NOW + 30},
                                 "aud-one"), None)
        self.assertEqual(outcome({"aud": ["aud-one"], "exp": NOW + 30},
                                 "aud-one"), None)
        self.assertEqual(outcome({"aud": ["aud-two"], "exp": NOW + 30},
                                 "aud-one"), "AudienceMismatchError")
        self.assertEqual(outcome({"aud": [], "exp": NOW + 30},
                                 "aud-one"), "AudienceMismatchError")


if __name__ == "__main__":
    unittest.main()
