import re
import ssl
import socket
import idna
import urllib.parse
from typing import Dict, Tuple, Optional, List
import aiohttp
import asyncio
import dns.asyncresolver
import whois
from datetime import datetime
import numpy as np
import time
from backend.schemas import URLAnalysis

# ── Constants ──────────────────────────────────────────────────────────────────

# Common TLDs for detection
COMMON_TLDS = [
    'com', 'org', 'net', 'edu', 'gov', 'mil', 'int',
    'info', 'biz', 'name', 'pro', 'aero', 'coop', 'museum',
    'co', 'io', 'me', 'tv', 'cc', 'us', 'uk', 'de', 'fr',
    'br', 'ru', 'cn', 'jp', 'kr', 'in', 'au', 'ca', 'id',
    'xyz', 'top', 'site', 'online', 'club', 'app', 'dev',
    'tech', 'store', 'shop', 'blog', 'cloud', 'ai', 'ly',
]

# URL shortener domains
URL_SHORTENERS = [
    'bit.ly', 'tinyurl.com', 't.co', 'goo.gl', 'is.gd', 'cli.gs',
    'ow.ly', 'rebrand.ly', 'bl.ink', 'short.io', 'cutt.ly',
    'buff.ly', 'soo.gd', 'tiny.cc', 'lnkd.in', 'db.tt',
    'qr.ae', 'adf.ly', 'bit.do', 'mcaf.ee', 'su.pr',
    'yourls.org', 'v.gd', 'tr.im',
]

# Default values matching training data medians for features
# that cannot be extracted in real-time
DEFAULTS = {
    'url_google_index': 0.0,       # median=0, most URLs indexed
    'domain_google_index': 0.0,    # median=0, most domains indexed
}

# ── Helper Functions ───────────────────────────────────────────────────────────

def _extract_root_domain(domain: str) -> str:
    """Extract the root domain (e.g., 'docs.github.com' -> 'github.com')."""
    parts = domain.split('.')
    if len(parts) <= 2:
        return domain
    # Handle country code TLDs like co.id, co.uk, com.br
    if len(parts) >= 3 and parts[-2] in ('co', 'com', 'org', 'net', 'ac', 'gov', 'edu'):
        return '.'.join(parts[-3:])
    return '.'.join(parts[-2:])


def process_punycode(url: str) -> Tuple[str, bool, Optional[str]]:
    """Decode Punycode and detect IDN homograph attack."""
    try:
        parsed = urllib.parse.urlparse(url)
        netloc = parsed.netloc
        if not netloc:
            return url, False, None
            
        decoded_netloc = idna.decode(netloc)
        
        is_punycode = decoded_netloc != netloc
        warning = None
        
        if is_punycode:
            # Check for non-ASCII characters that might be confusing
            has_non_ascii = any(ord(c) > 127 for c in decoded_netloc)
            if has_non_ascii:
                warning = "Domain mengandung karakter Unicode yang mencurigakan (IDN homograph suspect)."
                
        decoded_url = url.replace(netloc, decoded_netloc)
        return decoded_url, is_punycode, warning
        
    except Exception:
        return url, False, None


# ── URL Feature Extraction ─────────────────────────────────────────────────────

def extract_url_features(url: str) -> Tuple[Dict[str, float], str]:
    """Extract string parsing features from URL."""
    features = {}
    
    # ── URL-level character counts ──
    features['length_url'] = float(len(url))
    features['qty_dot_url'] = float(url.count('.'))
    features['qty_hyphen_url'] = float(url.count('-'))
    features['qty_underline_url'] = float(url.count('_'))
    features['qty_slash_url'] = float(url.count('/'))
    features['qty_questionmark_url'] = float(url.count('?'))
    features['qty_equal_url'] = float(url.count('='))
    features['qty_at_url'] = float(url.count('@'))
    features['qty_and_url'] = float(url.count('&'))
    features['qty_exclamation_url'] = float(url.count('!'))
    features['qty_space_url'] = float(url.count(' ') + url.count('%20'))
    features['qty_tilde_url'] = float(url.count('~'))
    features['qty_comma_url'] = float(url.count(','))
    features['qty_plus_url'] = float(url.count('+'))
    features['qty_asterisk_url'] = float(url.count('*'))
    features['qty_hashtag_url'] = float(url.count('#'))
    features['qty_dollar_url'] = float(url.count('$'))
    features['qty_percent_url'] = float(url.count('%'))
    
    # Count TLD occurrences in URL
    url_lower = url.lower()
    tld_count = 0
    for tld in COMMON_TLDS:
        tld_count += url_lower.count('.' + tld)
    features['qty_tld_url'] = float(max(1, tld_count))  # At least 1 TLD expected
    
    # Check for email in URL
    email_pattern = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
    features['email_in_url'] = 1.0 if email_pattern.search(url) else 0.0
    
    # Check if URL is shortened
    parsed = urllib.parse.urlparse(url)
    domain = parsed.netloc.lower()
    # Strip port if present
    if ':' in domain:
        domain = domain.split(':')[0]
    features['url_shortened'] = 1.0 if domain in URL_SHORTENERS else 0.0
    
    path = parsed.path
    
    # ── Domain-level features (ALL character counts) ──
    features['qty_dot_domain'] = float(domain.count('.'))
    features['qty_hyphen_domain'] = float(domain.count('-'))
    features['qty_underline_domain'] = float(domain.count('_'))
    features['qty_slash_domain'] = float(domain.count('/'))
    features['qty_questionmark_domain'] = float(domain.count('?'))
    features['qty_equal_domain'] = float(domain.count('='))
    features['qty_at_domain'] = float(domain.count('@'))
    features['qty_and_domain'] = float(domain.count('&'))
    features['qty_exclamation_domain'] = float(domain.count('!'))
    features['qty_space_domain'] = float(domain.count(' '))
    features['qty_tilde_domain'] = float(domain.count('~'))
    features['qty_comma_domain'] = float(domain.count(','))
    features['qty_plus_domain'] = float(domain.count('+'))
    features['qty_asterisk_domain'] = float(domain.count('*'))
    features['qty_hashtag_domain'] = float(domain.count('#'))
    features['qty_dollar_domain'] = float(domain.count('$'))
    features['qty_percent_domain'] = float(domain.count('%'))
    features['qty_vowels_domain'] = float(sum(1 for c in domain.lower() if c in 'aeiou'))
    features['domain_length'] = float(len(domain))
    features['domain_in_ip'] = 1.0 if re.match(
        r'^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$', domain
    ) else 0.0
    features['server_client_domain'] = 1.0 if 'server' in domain.lower() or 'client' in domain.lower() else 0.0
    
    # ── Path/Directory/File ──
    directory = path.rsplit('/', 1)[0] if '/' in path else ''
    file = path.rsplit('/', 1)[-1] if '/' in path else ''
    
    if len(directory) > 0 and directory != '/':
        features['qty_dot_directory'] = float(directory.count('.'))
        features['qty_hyphen_directory'] = float(directory.count('-'))
        features['qty_underline_directory'] = float(directory.count('_'))
        features['qty_slash_directory'] = float(directory.count('/'))
        features['qty_questionmark_directory'] = float(directory.count('?'))
        features['qty_equal_directory'] = float(directory.count('='))
        features['qty_at_directory'] = float(directory.count('@'))
        features['qty_and_directory'] = float(directory.count('&'))
        features['qty_exclamation_directory'] = float(directory.count('!'))
        features['qty_space_directory'] = float(directory.count(' ') + directory.count('%20'))
        features['qty_tilde_directory'] = float(directory.count('~'))
        features['qty_comma_directory'] = float(directory.count(','))
        features['qty_plus_directory'] = float(directory.count('+'))
        features['qty_asterisk_directory'] = float(directory.count('*'))
        features['qty_hashtag_directory'] = float(directory.count('#'))
        features['qty_dollar_directory'] = float(directory.count('$'))
        features['qty_percent_directory'] = float(directory.count('%'))
        features['directory_length'] = float(len(directory))
    
    if len(file) > 0 and file != '/':
        features['qty_dot_file'] = float(file.count('.'))
        features['qty_hyphen_file'] = float(file.count('-'))
        features['qty_underline_file'] = float(file.count('_'))
        features['qty_slash_file'] = float(file.count('/'))
        features['qty_questionmark_file'] = float(file.count('?'))
        features['qty_equal_file'] = float(file.count('='))
        features['qty_at_file'] = float(file.count('@'))
        features['qty_and_file'] = float(file.count('&'))
        features['qty_exclamation_file'] = float(file.count('!'))
        features['qty_space_file'] = float(file.count(' ') + file.count('%20'))
        features['qty_tilde_file'] = float(file.count('~'))
        features['qty_comma_file'] = float(file.count(','))
        features['qty_plus_file'] = float(file.count('+'))
        features['qty_asterisk_file'] = float(file.count('*'))
        features['qty_hashtag_file'] = float(file.count('#'))
        features['qty_dollar_file'] = float(file.count('$'))
        features['qty_percent_file'] = float(file.count('%'))
        features['file_length'] = float(len(file))

    # ── Parameters features ──
    params = parsed.query
    if params:
        features['qty_dot_params'] = float(params.count('.'))
        features['qty_hyphen_params'] = float(params.count('-'))
        features['qty_underline_params'] = float(params.count('_'))
        features['qty_slash_params'] = float(params.count('/'))
        features['qty_questionmark_params'] = float(params.count('?'))
        features['qty_equal_params'] = float(params.count('='))
        features['qty_at_params'] = float(params.count('@'))
        features['qty_and_params'] = float(params.count('&'))
        features['qty_exclamation_params'] = float(params.count('!'))
        features['qty_space_params'] = float(params.count(' ') + params.count('%20'))
        features['qty_tilde_params'] = float(params.count('~'))
        features['qty_comma_params'] = float(params.count(','))
        features['qty_plus_params'] = float(params.count('+'))
        features['qty_asterisk_params'] = float(params.count('*'))
        features['qty_hashtag_params'] = float(params.count('#'))
        features['qty_dollar_params'] = float(params.count('$'))
        features['qty_percent_params'] = float(params.count('%'))
        features['params_length'] = float(len(params))
        features['qty_params'] = float(len(urllib.parse.parse_qs(params)))
        
        # Check if TLD is present in params (suspicious)
        params_lower = params.lower()
        tld_in_params = any(('.' + tld) in params_lower for tld in COMMON_TLDS)
        features['tld_present_params'] = 1.0 if tld_in_params else -1.0
    else:
        # No params → match training data convention (-1 = absent)
        features['tld_present_params'] = -1.0

    return features, domain


# ── Network Feature Extraction ─────────────────────────────────────────────────

async def extract_network_features(url: str, domain: str) -> Dict[str, float]:
    """Extract network, DNS, and WHOIS features asynchronously with timeouts."""
    features = {}
    root_domain = _extract_root_domain(domain)
    
    # Run all network lookups concurrently
    http_task = _extract_http_features(url)
    dns_task = _extract_dns_features(domain, root_domain)
    whois_task = _extract_whois_features(root_domain)
    tls_task = _extract_tls_features(domain)
    
    http_feats, dns_feats, whois_feats, tls_feats = await asyncio.gather(
        http_task, dns_task, whois_task, tls_task,
        return_exceptions=True
    )
    
    # Merge results (handle exceptions gracefully)
    for result in [http_feats, dns_feats, whois_feats, tls_feats]:
        if isinstance(result, dict):
            features.update(result)
    
    # ── Defaults for features that can't be extracted real-time ──
    features.setdefault('url_google_index', DEFAULTS['url_google_index'])
    features.setdefault('domain_google_index', DEFAULTS['domain_google_index'])
    
    return features


async def _extract_http_features(url: str) -> Dict[str, float]:
    """Extract HTTP response features."""
    features = {}
    loop = asyncio.get_running_loop()
    def do_request():
        import requests
        start_time = time.perf_counter()
        try:
            r = requests.head(url, timeout=5, allow_redirects=True)
            dt = (time.perf_counter() - start_time) * 1000
            return {'time_response': float(dt), 'qty_redirects': float(len(r.history))}
        except Exception:
            return {'time_response': np.nan, 'qty_redirects': np.nan}
            
    try:
        features = await asyncio.wait_for(
            loop.run_in_executor(None, do_request),
            timeout=8.0
        )
    except Exception:
        features = {'time_response': np.nan, 'qty_redirects': np.nan}
        
    return features


async def _extract_dns_features(domain: str, root_domain: str) -> Dict[str, float]:
    """Extract DNS-related features (A, NS, MX, SPF)."""
    features = {}
    resolver = dns.asyncresolver.Resolver()
    resolver.timeout = 5
    resolver.lifetime = 5
    
    # ── A records (IP count + TTL) ──
    try:
        answers_a = await resolver.resolve(domain, 'A')
        features['qty_ip_resolved'] = float(len(answers_a))
        features['ttl_hostname'] = float(answers_a.rrset.ttl if answers_a.rrset else 300)
    except Exception:
        features['qty_ip_resolved'] = np.nan
        features['ttl_hostname'] = np.nan
        
    # ── NS records ──
    try:
        answers_ns = await resolver.resolve(root_domain, 'NS')
        features['qty_nameservers'] = float(len(answers_ns))
    except Exception:
        features['qty_nameservers'] = np.nan
    
    # ── MX records ──
    try:
        answers_mx = await resolver.resolve(root_domain, 'MX')
        features['qty_mx_servers'] = float(len(answers_mx))
    except Exception:
        features['qty_mx_servers'] = np.nan
        
    # ── SPF record (via TXT) ──
    try:
        answers_txt = await resolver.resolve(root_domain, 'TXT')
        has_spf = any('v=spf1' in str(rdata).lower() for rdata in answers_txt)
        features['domain_spf'] = 1.0 if has_spf else 0.0
    except Exception:
        features['domain_spf'] = np.nan
    
    return features


async def _extract_whois_features(root_domain: str) -> Dict[str, float]:
    """Extract WHOIS features (domain age, expiration)."""
    features = {}
    loop = asyncio.get_running_loop()
    
    def get_whois():
        try:
            return whois.whois(root_domain)
        except Exception:
            return None
            
    try:
        w = await asyncio.wait_for(
            loop.run_in_executor(None, get_whois), 
            timeout=8.0
        )
        
        if w and w.creation_date:
            creation = w.creation_date[0] if isinstance(w.creation_date, list) else w.creation_date
            if isinstance(creation, datetime):
                days_active = (datetime.now() - creation).days
                features['time_domain_activation'] = float(max(0, days_active))
            else:
                features['time_domain_activation'] = np.nan
        else:
            features['time_domain_activation'] = np.nan
            
        if w and w.expiration_date:
            expiration = w.expiration_date[0] if isinstance(w.expiration_date, list) else w.expiration_date
            if isinstance(expiration, datetime):
                days_exp = (expiration - datetime.now()).days
                features['time_domain_expiration'] = float(max(0, days_exp))
            else:
                features['time_domain_expiration'] = np.nan
        else:
            features['time_domain_expiration'] = np.nan
            
    except Exception:
        features['time_domain_activation'] = np.nan
        features['time_domain_expiration'] = np.nan
        
    return features


async def _extract_tls_features(domain: str) -> Dict[str, float]:
    """Extract TLS/SSL certificate features and ASN."""
    features = {}
    loop = asyncio.get_running_loop()
    
    # ── TLS Certificate check ──
    def check_tls():
        try:
            ctx = ssl.create_default_context()
            with ctx.wrap_socket(socket.socket(), server_hostname=domain) as s:
                s.settimeout(5)
                s.connect((domain, 443))
                cert = s.getpeercert()
                return 1.0 if cert else 0.0
        except Exception:
            return 0.0
    
    try:
        features['tls_ssl_certificate'] = await asyncio.wait_for(
            loop.run_in_executor(None, check_tls),
            timeout=6.0
        )
    except Exception:
        features['tls_ssl_certificate'] = 0.0
    
    # ── ASN lookup ──
    def get_asn():
        try:
            import socket as sock
            ip = sock.gethostbyname(domain)
            from ipwhois import IPWhois
            obj = IPWhois(ip)
            result = obj.lookup_rdap(depth=0)
            asn = result.get('asn', None)
            return float(asn) if asn and asn != 'NA' else np.nan
        except Exception:
            return np.nan
    
    try:
        features['asn_ip'] = await asyncio.wait_for(
            loop.run_in_executor(None, get_asn),
            timeout=8.0
        )
    except Exception:
        features['asn_ip'] = np.nan
    
    return features


# ── Main Extraction Pipeline ───────────────────────────────────────────────────

async def extract_all_features(url: str) -> Tuple[Dict[str, float], URLAnalysis]:
    """Main extraction pipeline combining synchronous parsing and async network calls."""
    # 1. Punycode
    from backend.schemas import URLAnalysis
    decoded_url, is_punycode, warning = process_punycode(url)
    
    analysis = URLAnalysis(
        url_original=url,
        url_decoded=decoded_url,
        is_punycode=is_punycode,
        punycode_warning=warning
    )
    
    # 2. Synchronous URL/Domain/Path Features
    url_features, domain = extract_url_features(decoded_url)
    
    # Add punycode features
    url_features['is_punycode'] = 1.0 if is_punycode else 0.0
    url_features['n_unicode_chars'] = float(len([c for c in decoded_url if ord(c) > 127]))
    url_features['homograph_score'] = 1.0 if is_punycode and warning else 0.0
    
    # 3. Asynchronous Network Features
    network_features = await extract_network_features(decoded_url, domain)
    
    # Merge
    all_features = {**url_features, **network_features}
    
    return all_features, analysis
