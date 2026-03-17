import re
import idna
import urllib.parse
from typing import Dict, Tuple, Optional
import aiohttp
import asyncio
import dns.asyncresolver
import whois
from datetime import datetime
import numpy as np
from backend.schemas import URLAnalysis

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

def extract_url_features(url: str) -> Dict[str, float]:
    """Extract string parsing features from URL."""
    features = {}
    
    # URL Structure
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
    
    parsed = urllib.parse.urlparse(url)
    domain = parsed.netloc
    path = parsed.path
    
    # Domain
    features['length_hostname'] = float(len(domain))
    features['ip_present'] = 1.0 if re.match(r'[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}', domain) else 0.0
    features['https_present'] = 1.0 if parsed.scheme == 'https' else 0.0
    features['qty_dot_domain'] = float(domain.count('.'))
    features['qty_hyphen_domain'] = float(domain.count('-'))
    features['qty_vowels_domain'] = float(sum(1 for c in domain.lower() if c in 'aeiou'))
    features['domain_length'] = float(len(domain))
    features['server_client_domain'] = 1.0 if 'server' in domain.lower() or 'client' in domain.lower() else 0.0
    
    # Path/Directory/File
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

    # Add parameters features if present
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

    return features, domain

async def extract_network_features(url: str, domain: str) -> Dict[str, float]:
    """Extract network, DNS, and WHOIS features asynchronously with timeouts."""
    features = {}
    
    # 1. HTTP Request (Response Time, Redirects)
    start_time = asyncio.get_event_loop().time()
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
            async with session.head(url, allow_redirects=True) as response:
                dt = (asyncio.get_event_loop().time() - start_time) * 1000
                features['time_response'] = float(dt)
                features['qty_redirects'] = float(len(response.history))
    except Exception:
        features['time_response'] = np.nan
        features['qty_redirects'] = np.nan

    # 2. DNS Resolution
    try:
        resolver = dns.asyncresolver.Resolver()
        resolver.timeout = 3
        resolver.lifetime = 3
        
        # A records (IPs)
        try:
            answers_a = await resolver.resolve(domain, 'A')
            features['qty_ip_resolved'] = float(len(answers_a))
            features['ttl_hostname'] = float(answers_a.rrset.ttl if answers_a.rrset else FALLBACK_TTL_HOSTNAME)
        except Exception:
            features['qty_ip_resolved'] = np.nan
            features['ttl_hostname'] = np.nan
            
        # NS records
        try:
            answers_ns = await resolver.resolve(domain, 'NS')
            features['qty_nameservers'] = float(len(answers_ns))
        except Exception:
            features['qty_nameservers'] = np.nan
            
    except Exception:
        features['qty_ip_resolved'] = np.nan
        features['ttl_hostname'] = np.nan
        features['qty_nameservers'] = np.nan

    # 3. WHOIS (Synchronous, but wrapped in executor)
    loop = asyncio.get_running_loop()
    def get_whois():
        try:
            return whois.whois(domain)
        except Exception:
            return None
            
    try:
        w = await asyncio.wait_for(loop.run_in_executor(None, get_whois), timeout=5.0)
        
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
