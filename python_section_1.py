from typing import Dict, List

import pandas as pd


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    # Your code goes here.
    
    def reverse_in_groups(lst, n):
    result = []
    length = len(lst)
    
    for i in range(0, length, n):
        group_end = min(i + n, length)
        group = lst[i:group_end]  # Grab the group
        
        # Manually reverse the group
        reversed_group = []
        for j in range(len(group)):
            reversed_group.append(group[len(group) - 1 - j])
        
        result.extend(reversed_group)  # Add the reversed group to the result
    
    return result

# Example usage
print(reverse_in_groups([1, 2, 3, 4, 5, 6, 7, 8], 3))  # Output: [3, 2, 1, 6, 5, 4, 8, 7]
print(reverse_in_groups([1, 2, 3, 4, 5], 2))           # Output: [2, 1, 4, 3, 5]
print(reverse_in_groups([10, 20, 30, 40, 50, 60, 70], 4))  # Output: [40, 30, 20, 10, 70, 60, 50]


    


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    # Your code here


    def group_by_length(strings):
    length_dict = {}

    for string in strings:
        length = len(string)
        if length not in length_dict:
            length_dict[length] = []
        length_dict[length].append(string)

    # Sort the dictionary by keys (lengths) and return it
    sorted_length_dict = dict(sorted(length_dict.items()))

    return sorted_length_dict

# Example usage
print(group_by_length(["apple", "bat", "car", "elephant", "dog", "bear"]))
# Output: {3: ['bat', 'car', 'dog'], 4: ['bear'], 5: ['apple'], 8: ['elephant']}

print(group_by_length(["one", "two", "three", "four"]))
# Output: {3: ['one', 'two'], 4: ['four'], 5: ['three']}




def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    # Your code here
  def flatten_dict(nested_dict, parent_key='', sep='.'):
    items = {}
    
    for key, value in nested_dict.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        
        if isinstance(value, dict):
            # Recursive call for nested dictionaries
            items.update(flatten_dict(value, new_key, sep=sep))
        elif isinstance(value, list):
            # Handle lists by including index in the key
            for i, item in enumerate(value):
                list_key = f"{new_key}[{i}]"
                if isinstance(item, dict):
                    # Recursive call if list item is a dictionary
                    items.update(flatten_dict(item, list_key, sep=sep))
                else:
                    items[list_key] = item
        else:
            # Base case for regular key-value pairs
            items[new_key] = value

    return items

# Example usage
nested_dict = {
    'a': 1,
    'b': {
        'c': 2,
        'd': {
            'e': 3,
            'f': [4, 5, {'g': 6}]
        }
    },
    'h': [7, 8]
}

flattened = flatten_dict(nested_dict)
print(flattened)


def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    # Your code here
   def unique_permutations(nums):
    def backtrack(start):
        # If we've reached the end of the array, we've found a valid permutation
        if start == len(nums):
            result.append(nums[:])  # Append a copy of the current permutation
            return
        
        seen = set()  # To keep track of used elements at this level of recursion
        for i in range(start, len(nums)):
            if nums[i] in seen:
                continue  # Skip duplicates
            
            seen.add(nums[i])  # Mark the number as seen
            # Swap the current element with the starting element
            nums[start], nums[i] = nums[i], nums[start]
            # Recurse with the next element
            backtrack(start + 1)
            # Backtrack: swap back
            nums[start], nums[i] = nums[i], nums[start]

    result = []
    nums.sort()  # Sort to ensure duplicates are adjacent
    backtrack(0)
    return result

# Example usage
input_list = [1, 1, 2]
output = unique_permutations(input_list)
print(output)



def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    import re

def find_all_dates(text):
    # Define regex patterns for the date formats
    patterns = [
        r'\b\d{2}-\d{2}-\d{4}\b',  # dd-mm-yyyy
        r'\b\d{2}/\d{2}/\d{4}\b',  # mm/dd/yyyy
        r'\b\d{4}\.\d{2}\.\d{2}\b'  # yyyy.mm.dd
    ]
    
    # Combine patterns into a single regex
    combined_pattern = '|'.join(patterns)
    
    # Find all matches in the text
    matches = re.findall(combined_pattern, text)
    
    return matches

# Example usage
text = "I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."
output = find_all_dates(text)
print(output)


def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
   import polyline
import pandas as pd
import numpy as np

def haversine(lat1, lon1, lat2, lon2):
    # Haversine formula to calculate the distance between two points
    R = 6371000  # Radius of the Earth in meters
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    
    a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c  # Distance in meters

def decode_polyline(polyline_str):
    # Decode the polyline string
    coordinates = polyline.decode(polyline_str)
    
    # Create a DataFrame
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
    
    # Calculate distances
    distances = [0]  # First point distance is 0
    for i in range(1, len(df)):
        distance = haversine(df.iloc[i-1]['latitude'], df.iloc[i-1]['longitude'],
                             df.iloc[i]['latitude'], df.iloc[i]['longitude'])
        distances.append(distance)
    
    df['distance'] = distances
    return df

# Example usage
polyline_str = "a~l~Fjk~u@_@_@_@_@"
df = decode_polyline(polyline_str)
print(df)



def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    # Your code here
   def rotate_matrix(matrix):
    n = len(matrix)
    
    # Step 1: Rotate the matrix by 90 degrees clockwise
    rotated_matrix = [[matrix[n - j - 1][i] for j in range(n)] for i in range(n)]
    
    # Step 2: Transform the rotated matrix
    final_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated_matrix[i])
            col_sum = sum(rotated_matrix[k][j] for k in range(n))
            # Replace with the sum of row and column, excluding itself
            final_matrix[i][j] = row_sum + col_sum - rotated_matrix[i][j]

    return final_matrix

# Example usage
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
result = rotate_matrix(matrix)
print(result)



def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here

    import pandas as pd
import numpy as np

def check_time_coverage(df):
    # First, parse start and end times into datetime formats
    df['start_datetime'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'], format='%Y-%m-%d %H:%M:%S')
    df['end_datetime'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'], format='%Y-%m-%d %H:%M:%S')

    # Create a new column to track the day of the week from the 'startDay'
    df['start_day_of_week'] = df['start_datetime'].dt.dayofweek
    df['end_day_of_week'] = df['end_datetime'].dt.dayofweek

    # Define full week and full day coverage
    full_week = set(range(7))  # 0 is Monday, 6 is Sunday
    full_day = pd.date_range('00:00:00', '23:59:59', freq='1S')

    # Group by (id, id_2) and check coverage
    def verify_coverage(group):
        # Extract covered days
        covered_days = set(group['start_day_of_week']) | set(group['end_day_of_week'])
        
        # For time, concatenate all periods within the group
        time_coverage = pd.Series(dtype='bool')
        for _, row in group.iterrows():
            time_range = pd.date_range(row['start_datetime'], row['end_datetime'], freq='1S')
            time_coverage = time_coverage.append(pd.Series(time_range))

        # Remove duplicates from time coverage and sort
        time_coverage = time_coverage.drop_duplicates().sort_values()

        # Check if all days and full time are covered
        is_full_week = covered_days == full_week
        is_full_day = time_coverage.equals(full_day)

        # Return True if both conditions are satisfied, False otherwise
        return is_full_week and is_full_day

    # Apply the function to each (id, id_2) group
    result = df.groupby(['id', 'id_2']).apply(verify_coverage)

    # Return the result as a boolean Series with multi-index
    return result

