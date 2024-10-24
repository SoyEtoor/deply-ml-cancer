import requests

body = {
    "diagnosis": 569.0,
    "radius_mean": 17.99,
    "texture_mean": 10.38,
    "perimeter_mean": 122.80,
    "area_mean": 1001.0,
    "smoothness_mean": 0.11840,
    "compactness_mean": 0.27760,
    "concavity_mean": 0.3001,
    "concave_points_mean": 0.14710,
    "symmetry_mean": 0.2419,
    "fractal_dimension_mean": 0.07864,
    "radius_se": 1.095,
    "texture_se": 0.9053,
    "perimeter_se": 8.589,
    "area_se": 153.4,
    "smoothness_se": 0.00645,
    "compactness_se": 0.04904,
    "concavity_se": 0.05373,
    "concave_points_se": 0.01587,
    "symmetry_se": 0.03022,
    "fractal_dimension_se": 0.00615,
    "radius_worst": 25.38,
    "texture_worst": 17.33,
    "perimeter_worst": 184.60,
    "area_worst": 2019.0,
    "smoothness_worst": 0.1622,
    "compactness_worst": 0.6656,
    "concavity_worst": 0.7119,
    "concave_points_worst": 0.2654,
    "symmetry_worst": 0.4601,
    "fractal_dimension_worst": 0.11890
}

response = requests.post(url='http://127.0.0.1:8000/score', json=body)
print(response.json())
