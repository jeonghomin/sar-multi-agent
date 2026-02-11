"""
Module to convert location names to coordinates using Google Maps API
"""

import requests
import time
import re
import os
from typing import Optional, Tuple, List
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv(override=True)

# Google Maps API key (환경 변수에서 가져오거나 기본값 사용)
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "AIzaSyAzsD6eYolLgTIW3SnEFTypyBxVlqBCXX0")


def get_coordinates_googlemaps(location_name, api_key=GOOGLE_MAPS_API_KEY, verbose=True):
    """
    Convert a location name to latitude and longitude using Google Maps Geocoding API.
    
    Args:
        location_name (str): Location name or address
        api_key (str): Google Maps API key
        verbose (bool, optional): Whether to print detailed information (default: True)
        
    Returns:
        tuple: (latitude, longitude, formatted_address) or (None, None, None) on error
    """
    try:
        # API request URL and parameters
        url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {
            'address': location_name,
            'key': api_key
        }
        
        # Send API request
        response = requests.get(url, params=params)
        result = response.json()

        # Check request success and results
        if result['status'] == 'OK':
            # Use first result
            location = result['results'][0]
            lat = location['geometry']['location']['lat']
            lng = location['geometry']['location']['lng']
            formatted_address = location['formatted_address']
            
            if verbose:
                print(f"Address: {formatted_address}")
                print(f"Latitude: {lat}")
                print(f"Longitude: {lng}")
            
            return (lat, lng, formatted_address)
        else:
            if verbose:
                print(f"Google Maps API error: {result['status']}")
            
            # Retry with variants if API error occurs
            # Generate location variants
            variants = generate_location_variants(location_name)
            
            # Skip first one as it's the original
            for variant in variants[1:] if len(variants) > 1 else []:
                if verbose:
                    print(f"Trying variant location: '{variant}'")
                
                # Retry coordinate search with variant location
                retry_result = get_coordinates_googlemaps(variant, api_key, verbose=verbose)
                if retry_result[0] is not None:
                    if verbose:
                        print(f"Coordinates found for variant location '{variant}'")
                    return retry_result
            
            # All variant attempts failed
            return (None, None, None)
            
    except Exception as e:
        if verbose:
            print(f"Error Searching Coordinates: {e}")
        return (None, None, None)


def get_coordinates_from_related_terms(related_terms, target_location=None, verbose=True):
    """
    Extract location information from a list of related search terms.
    
    Args:
        related_terms (list): List of related search terms
        target_location (str, optional): Original extracted location name (used for comparison if provided)
        verbose (bool): Whether to print detailed information
        
    Returns:
        tuple: (latitude, longitude, formatted_address) or (None, None, None) on error
    """
    if not related_terms:
        if verbose:
            print("No related search terms available.")
        return (None, None, None)
    
    if verbose:
        print(f"Trying related search terms as location information: {', '.join(related_terms)}")
    
    # If original location name exists, prioritize matching or containing items by comparing with related terms
    prioritized_terms = []
    if target_location:
        target_lower = target_location.lower()
        target_words = set(word.lower() for word in target_location.replace(',', ' ').split() if len(word) > 2)
        
        # 1. Exactly matching items
        for term in related_terms:
            if term.lower() == target_lower:
                prioritized_terms.append(term)
                if verbose:
                    print(f"Found related search term exactly matching original location: '{term}'")
        
        # 2. Items containing original location name
        if not prioritized_terms:
            for term in related_terms:
                if target_lower in term.lower():
                    prioritized_terms.append(term)
                    if verbose:
                        print(f"Found related search term containing original location name: '{term}'")
        
        # 3. Items containing many key words from original location
        if not prioritized_terms:
            term_scores = []
            for term in related_terms:
                term_lower = term.lower()
                matching_words = sum(1 for word in target_words if word in term_lower)
                if matching_words > 0:
                    term_scores.append((term, matching_words))
            
            # Sort by number of matching words
            term_scores.sort(key=lambda x: x[1], reverse=True)
            for term, score in term_scores:
                prioritized_terms.append(term)
                if verbose:
                    print(f"Found related search term with {score} matching words from original location: '{term}'")
    
    # Use original list if no priority terms
    search_terms = prioritized_terms + [term for term in related_terms if term not in prioritized_terms]
    
    # Try each search term as location information
    for term in search_terms:
        if verbose:
            print(f"Trying related search term: '{term}'")
        
        # Search coordinates with current search term
        result = get_coordinates_googlemaps(term, verbose=verbose)
        
        # Return if successful
        if result[0] is not None:
            if verbose:
                print(f"Coordinates found for related search term '{term}'")
            return result
    
    if verbose:
        print("Could not find coordinates from any related search terms.")
    return (None, None, None)


def generate_location_variants(location_name):
    """
    Generate various variants of a location name.
    
    Args:
        location_name (str): Location name or address
    
    Returns:
        list: List of various variants of the location name
    """
    variants = [location_name]  # Include original location name
    
    # Split by comma and whitespace
    components = [comp.strip() for comp in location_name.split(',')]
    components = [comp for comp in components if comp]  # Remove empty elements
    
    # Check if location name is university-related
    university_keywords = ['university', 'college', 'institute', 'school', 'campus']
    is_university = any(keyword in location_name.lower() for keyword in university_keywords)
    
    # Extract zip code (e.g., 12345 or 12345-6789)
    zip_match = re.search(r'\b(\d{5}(-\d{4})?)\b', location_name)
    zip_code = zip_match.group(1) if zip_match else None
    
    # Generate component-based variants
    if len(components) >= 2:
        # First part only (usually landmark/institution name)
        variants.append(components[0])
        
        # First two parts (usually landmark+city)
        variants.append(f"{components[0]}, {components[1]}")
        
        # First three parts (usually landmark+city+state)
        if len(components) >= 3:
            variants.append(f"{components[0]}, {components[1]}, {components[2]}")
    
    # Additional variants for university-related locations
    if is_university:
        # Try to extract university name only
        university_parts = []
        for comp in components:
            for keyword in university_keywords:
                if keyword in comp.lower():
                    university_parts.append(comp)
                    break
        
        if university_parts:
            for part in university_parts:
                if part not in variants:
                    variants.append(part)
                
                # Add university+city combination if city or state exists
                for other_comp in components:
                    if other_comp != part and other_comp not in university_parts:
                        combined = f"{part}, {other_comp}"
                        if combined not in variants:
                            variants.append(combined)
    
    # Abbreviated variants for long location names
    words = location_name.split()
    if len(words) > 4:
        variants.append(' '.join(words[:4]))
        variants.append(' '.join(words[:3]))
    
    # Remove duplicates and return
    unique_variants = []
    for variant in variants:
        if variant and variant not in unique_variants:
            unique_variants.append(variant)
    
    return unique_variants


def get_coordinates_with_retry(location, try_variants=False, verbose=True):
    """
    Extract coordinates from given location name. Try multiple variants on failure.
    
    Args:
        location (str): Location name
        try_variants (bool): Whether to try location name variants
        verbose (bool): Whether to print detailed information
        
    Returns:
        tuple: (latitude, longitude, formatted_address) or (None, None, None) on error
    """
    if verbose:
        print(f"Analyzing location: '{location}'")
    
    # Remove parentheses and their contents from location information
    clean_location = re.sub(r'\([^)]*\)', '', location).strip()
    
    # Show if original and cleaned location names differ
    if clean_location != location and verbose:
        print(f"Removed parentheses: '{location}' -> '{clean_location}'")
        location = clean_location
    
    # Try original location name first
    lat, lng, address = get_coordinates_googlemaps(location, verbose=verbose)
    
    # Return immediately if successful
    if lat is not None and lng is not None:
        if verbose:
            print(f"Found coordinates for original location name")
        return lat, lng, address
    
    # If failed and variant retry option is enabled
    if try_variants:
        # Generate location name variants
        variants = generate_location_variants(location)
        
        if verbose:
            print(f"Generated {len(variants)} location variants")
            for i, variant in enumerate(variants, 1):
                print(f"   {i}. '{variant}'")
        
        # Try each variant
        for variant in variants:
            if verbose:
                print(f"Trying variant: '{variant}'")
            
            lat, lng, address = get_coordinates_googlemaps(variant, verbose=verbose)
            
            # Return immediately if successful
            if lat is not None and lng is not None:
                if verbose:
                    print(f"Found coordinates for variant: {variant}")
                return lat, lng, address
    
    # All attempts failed
    if verbose:
        print(f"Could not find coordinates for location: {location}")
    
    return None, None, None


def location_to_coordinates(location_name: str, try_variants: bool = True, verbose: bool = True) -> Optional[dict]:
    """
    지역명을 좌표(위도, 경도)로 변환합니다. (nodes.py에서 사용하는 형식)
    
    Args:
        location_name: 지역명 (예: "샌프란시스코", "San Francisco")
        try_variants: 변형된 지역명도 시도할지 여부
        verbose: 상세 정보 출력 여부
        
    Returns:
        {"latitude": float, "longitude": float, "location": str} 또는 None
    """
    lat, lng, address = get_coordinates_with_retry(location_name, try_variants=try_variants, verbose=verbose)
    
    if lat is not None and lng is not None:
        return {
            "latitude": lat,
            "longitude": lng,
            "location": address
        }
    return None


def extract_locations_from_text(text: str) -> List[str]:
    """
    텍스트에서 지역명을 추출합니다.
    간단한 휴리스틱: 대문자로 시작하는 단어나 일반적인 지역명 패턴 찾기
    """
    locations = []
    
    # 대문자로 시작하는 단어 찾기 (간단한 휴리스틱)
    words = re.findall(r'\b[A-Z][a-z]+\b', text)
    locations.extend(words)
    
    # 한국 도시명 패턴
    korean_cities = re.findall(r'[가-힣]+시|[가-힣]+도|[가-힣]+군|[가-힣]+구', text)
    locations.extend(korean_cities)
    
    return list(set(locations))  # 중복 제거

