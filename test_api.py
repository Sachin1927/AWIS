import requests
import time

API_URL = "http://localhost:8000"

print("Testing AWIS API Connection...")
print("="*50)

# Test 1: Health check
print("\n1. Testing /health endpoint...")
try:
    start = time.time()
    response = requests.get(f"{API_URL}/health", timeout=60)
    elapsed = time.time() - start
    
    if response.status_code == 200:
        print(f"✅ Health check OK ({elapsed:.2f}s)")
        print(f"   Response: {response.json()}")
    else:
        print(f"❌ Health check failed: {response.status_code}")
except Exception as e:
    print(f"❌ Connection failed: {e}")

# Test 2: Login
print("\n2. Testing /auth/login endpoint...")
try:
    start = time.time()
    response = requests.post(
        f"{API_URL}/auth/login",
        data={"username": "admin", "password": "admin123"},
        timeout=60
    )
    elapsed = time.time() - start
    
    if response.status_code == 200:
        print(f"✅ Login OK ({elapsed:.2f}s)")
        token = response.json()['access_token']
        print(f"   Token: {token[:30]}...")
        
        # Test 3: Get stats
        print("\n3. Testing /attrition/stats endpoint...")
        headers = {"Authorization": f"Bearer {token}"}
        start = time.time()
        response = requests.get(f"{API_URL}/attrition/stats", headers=headers, timeout=60)
        elapsed = time.time() - start
        
        if response.status_code == 200:
            print(f"✅ Stats OK ({elapsed:.2f}s)")
            stats = response.json()
            print(f"   Total Employees: {stats['total_employees']}")
            print(f"   High Risk: {stats['high_risk_count']}")
        else:
            print(f"❌ Stats failed: {response.status_code}")
    else:
        print(f"❌ Login failed: {response.status_code}")
except Exception as e:
    print(f"❌ Login failed: {e}")

print("\n" + "="*50)
print("API diagnostics complete!")