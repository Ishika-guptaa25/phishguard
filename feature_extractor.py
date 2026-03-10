import re
from urllib.parse import urlparse
import numpy as np


class PhishingFeatureExtractor:
    """
    Extract features from URLs to detect phishing patterns.
    Analyzes 30+ features including URL structure, character patterns, and domain properties.
    """

    def __init__(self):
        self.suspicious_keywords = [
            'paypal', 'amazon', 'apple', 'microsoft', 'google', 'facebook',
            'login', 'signin', 'account', 'update', 'verify', 'confirm',
            'urgent', 'action', 'secure', 'click', 'limited', 'expired',
            'suspended', 'locked', 'unusual', 'activity', 'validate'
        ]

    def extract_features(self, url):
        """Extract all features from a URL"""
        features = {}

        # 1. URL Length features
        features['url_length'] = len(url)
        features['url_length_encoded'] = len(url.encode('utf-8'))

        # 2. Protocol features
        features['has_http'] = 1 if url.startswith('http://') else 0
        features['has_https'] = 1 if url.startswith('https://') else 0
        features['protocol_matches_domain'] = self._check_protocol_domain_match(url)

        # 3. Domain and host features
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            path = parsed.path
            query = parsed.query
            fragment = parsed.fragment

            features['domain_length'] = len(domain)
            features['path_length'] = len(path)
            features['query_length'] = len(query)
            features['fragment_length'] = len(fragment)

            # 4. Dot and slash patterns
            features['dots_in_domain'] = domain.count('.')
            features['slashes_in_url'] = url.count('/')
            features['double_slash_redirect'] = 1 if '//' in path else 0

            # 5. Special character features
            features['at_sign_count'] = url.count('@')
            features['hyphen_in_domain'] = domain.count('-')
            features['underscore_in_domain'] = domain.count('_')
            features['hyphen_in_url'] = url.count('-')
            features['underscore_in_url'] = url.count('_')

            # 6. Suspicious pattern detection
            features['has_suspicious_keywords'] = self._count_suspicious_keywords(url)
            features['has_ip_address'] = self._has_ip_address(domain)
            features['has_digits_in_domain'] = self._has_digits_in_domain(domain)
            features['has_www'] = 1 if domain.startswith('www') else 0

            # 7. TLD and domain structure
            tld = self._get_tld(domain)
            features['tld_length'] = len(tld)
            features['domain_parts'] = domain.count('.') + 1
            features['has_unusual_tld'] = 1 if tld in self._get_unusual_tlds() else 0

            # 8. URL encoding and character distribution
            features['percent_encoding'] = url.count('%')
            features['uppercase_letters'] = sum(1 for c in url if c.isupper())
            features['lowercase_letters'] = sum(1 for c in url if c.islower())
            features['digits_in_url'] = sum(1 for c in url if c.isdigit())

            # 9. Port and protocol anomalies
            features['has_port'] = 1 if ':' in domain else 0
            if ':' in domain:
                try:
                    port = int(domain.split(':')[1])
                    features['suspicious_port'] = 1 if port not in [80, 443] else 0
                except:
                    features['suspicious_port'] = 0
            else:
                features['suspicious_port'] = 0

            # 10. Query string features
            features['has_query_string'] = 1 if query else 0
            features['query_parameter_count'] = query.count('=')
            features['has_url_shortener'] = 1 if self._is_url_shortener(domain) else 0

            # 11. Subdomain analysis
            subdomains = domain.split('.')
            features['subdomain_count'] = len(subdomains) - 1 if '.' in domain else 0
            features['suspicious_subdomain_pattern'] = self._check_suspicious_subdomains(domain)

        except Exception as e:
            # Return default features on parsing error
            for key in [
                'domain_length', 'path_length', 'query_length', 'fragment_length',
                'dots_in_domain', 'slashes_in_url', 'double_slash_redirect',
                'at_sign_count', 'hyphen_in_domain', 'underscore_in_domain',
                'hyphen_in_url', 'underscore_in_url', 'has_suspicious_keywords',
                'has_ip_address', 'has_digits_in_domain', 'has_www', 'tld_length',
                'domain_parts', 'has_unusual_tld', 'percent_encoding',
                'uppercase_letters', 'lowercase_letters', 'digits_in_url',
                'has_port', 'suspicious_port', 'has_query_string',
                'query_parameter_count', 'has_url_shortener', 'subdomain_count',
                'suspicious_subdomain_pattern'
            ]:
                features[key] = 0

        # Ensure consistent feature order
        feature_names = sorted(features.keys())
        return np.array([features[name] for name in feature_names]), feature_names

    @staticmethod
    def _check_protocol_domain_match(url):
        """Check if protocol and domain match safely"""
        try:
            if url.startswith('https://') and 'http://' in url[8:]:
                return 0
            return 1
        except:
            return 0

    def _count_suspicious_keywords(self, url):
        """Count occurrences of suspicious keywords"""
        url_lower = url.lower()
        count = 0
        for keyword in self.suspicious_keywords:
            count += url_lower.count(keyword)
        return min(count, 5)  # Cap at 5

    @staticmethod
    def _has_ip_address(domain):
        """Check if domain is an IP address"""
        pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
        return 1 if re.match(pattern, domain.split(':')[0]) else 0

    @staticmethod
    def _has_digits_in_domain(domain):
        """Check if domain contains digits"""
        return 1 if any(c.isdigit() for c in domain.split('.')[0]) else 0

    @staticmethod
    def _get_tld(domain):
        """Extract TLD from domain"""
        parts = domain.split('.')
        return parts[-1] if parts else ''

    @staticmethod
    def _get_unusual_tlds():
        """List of unusual/suspicious TLDs"""
        return {
            'tk', 'ml', 'ga', 'cf', 'zip', 'loan', 'download', 'stream',
            'win', 'top', 'pw', 'link', 'site', 'bid', 'click', 'date',
            'men', 'movie', 'ren', 'review', 'ru', 'xyz', 'gq'
        }

    @staticmethod
    def _is_url_shortener(domain):
        """Check if domain is a URL shortener"""
        shorteners = {'bit.ly', 'tinyurl.com', 'goo.gl', 'ow.ly', 'short.link'}
        return 1 if domain in shorteners else 0

    @staticmethod
    def _check_suspicious_subdomains(domain):
        """Check for suspicious subdomain patterns"""
        subdomains = domain.split('.')
        suspicious_count = 0
        for subdomain in subdomains[:-1]:
            if any(char in subdomain for char in ['-', '_']) or len(subdomain) < 3:
                suspicious_count += 1
        return min(suspicious_count, 3)