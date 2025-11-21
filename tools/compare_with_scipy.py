#!/usr/bin/env python3
"""
Scipy comparison test for logm implementation
Compares mathjs logm results with scipy.linalg.logm for various test matrices
"""

import numpy as np
from scipy.linalg import logm as scipy_logm, expm
import json
import subprocess
import sys

def test_matrix(name, A, description=""):
    """Test a matrix against scipy"""
    print(f"\n{'='*60}")
    print(f"Test: {name}")
    if description:
        print(f"Description: {description}")
    print(f"{'='*60}")
    
    # Compute scipy logm
    scipy_result = scipy_logm(A)
    
    # Create mathjs test
    mathjs_code = f"""
const math = require('./lib/cjs/defaultInstance.js').default
const A = {json.dumps(A.tolist())}
try {{
  const result = math.logm(A)
  const resultArray = result.valueOf ? result.valueOf() : result
  console.log(JSON.stringify({{success: true, result: resultArray}}))
}} catch(e) {{
  console.log(JSON.stringify({{success: false, error: e.message}}))
}}
"""
    
    # Run mathjs
    try:
        result = subprocess.run(
            ['node', '-e', mathjs_code],
            cwd='/home/runner/work/mathjs/mathjs',
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0:
            print(f"❌ Mathjs error: {result.stderr}")
            return None
            
        mathjs_result_data = json.loads(result.stdout)
        
        if not mathjs_result_data['success']:
            print(f"❌ Mathjs error: {mathjs_result_data['error']}")
            return None
            
        mathjs_result = np.array(mathjs_result_data['result'])
        
        # Compare results
        diff = np.abs(mathjs_result - scipy_result)
        max_diff = np.max(diff)
        rel_error = max_diff / (np.max(np.abs(scipy_result)) + 1e-15)
        
        print(f"\nScipy result:\n{scipy_result}")
        print(f"\nMathjs result:\n{mathjs_result}")
        print(f"\nMax absolute difference: {max_diff:.2e}")
        print(f"Max relative error: {rel_error:.2e}")
        
        # Check roundtrip: expm(logm(A)) ≈ A
        scipy_roundtrip = expm(scipy_result)
        mathjs_roundtrip = expm(mathjs_result)
        
        scipy_roundtrip_error = np.max(np.abs(scipy_roundtrip - A)) / (np.max(np.abs(A)) + 1e-15)
        mathjs_roundtrip_error = np.max(np.abs(mathjs_roundtrip - A)) / (np.max(np.abs(A)) + 1e-15)
        
        print(f"\nRoundtrip expm(logm(A)) ≈ A:")
        print(f"  Scipy relative error: {scipy_roundtrip_error:.2e}")
        print(f"  Mathjs relative error: {mathjs_roundtrip_error:.2e}")
        
        # Suggested tolerance is max of direct comparison and roundtrip
        suggested_tol = max(rel_error, mathjs_roundtrip_error) * 1.2  # 20% safety margin
        
        print(f"\n📊 Suggested tolerance: {suggested_tol:.3f}")
        
        if suggested_tol > 0.1:
            print(f"⚠️  HIGH ERROR - investigating...")
            print(f"   This may indicate an implementation issue")
        elif suggested_tol > 0.01:
            print(f"⚠️  Moderate error - acceptable for challenging cases")
        else:
            print(f"✅ Good accuracy")
            
        return {
            'name': name,
            'max_diff': float(max_diff),
            'rel_error': float(rel_error),
            'mathjs_roundtrip_error': float(mathjs_roundtrip_error),
            'scipy_roundtrip_error': float(scipy_roundtrip_error),
            'suggested_tolerance': float(suggested_tol)
        }
        
    except subprocess.TimeoutExpired:
        print(f"❌ Timeout")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    print("Scipy vs Mathjs logm Comparison Test")
    print("="*60)
    
    results = []
    
    # Test 1: Simple 2x2
    A1 = np.array([[1, 2], [3, 4]], dtype=float)
    r1 = test_matrix("2x2 simple", A1, "Basic 2x2 matrix")
    if r1: results.append(r1)
    
    # Test 2: Diagonal
    A2 = np.array([[np.e, 0], [0, np.e**2]], dtype=float)
    r2 = test_matrix("2x2 diagonal", A2, "Diagonal with e and e^2")
    if r2: results.append(r2)
    
    # Test 3: Rotation matrix
    theta = np.pi / 4
    A3 = np.array([[np.cos(theta), -np.sin(theta)], 
                   [np.sin(theta), np.cos(theta)]], dtype=float)
    r3 = test_matrix("Rotation matrix", A3, "π/4 rotation (complex eigenvalues)")
    if r3: results.append(r3)
    
    # Test 4: 3x3 upper triangular (current tolerance 0.4 = 40%)
    A4 = np.array([[1, 2, 3], [0, 1, 4], [0, 0, 1]], dtype=float)
    r4 = test_matrix("3x3 upper triangular", A4, "Current tolerance: 0.4 (40%)")
    if r4: results.append(r4)
    
    # Test 5: 5x5 with complex eigenvalues (current tolerance 1.5 = 150%)
    A5 = np.array([
        [1, 2, 0, 0, 0],
        [-2, 1, 1, 0, 0],
        [0, 0, 3, 0, 0],
        [0, 0, 0, 2, 1],
        [0, 0, 0, -1, 2]
    ], dtype=float)
    r5 = test_matrix("5x5 mixed blocks", A5, "Current tolerance: 1.5 (150%)")
    if r5: results.append(r5)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if results:
        print(f"\nCompleted {len(results)} tests\n")
        print(f"{'Test':<25} {'Current':<10} {'Suggested':<10} {'Status'}")
        print("-"*60)
        
        # Map test names to current tolerances
        current_tols = {
            '2x2 simple': 0.01,
            '2x2 diagonal': 0.01,
            'Rotation matrix': 0.01,
            '3x3 upper triangular': 0.4,
            '5x5 mixed blocks': 1.5
        }
        
        for r in results:
            current = current_tols.get(r['name'], 0.01)
            suggested = r['suggested_tolerance']
            status = "✅" if suggested < current * 0.8 else ("⚠️" if suggested > current else "OK")
            print(f"{r['name']:<25} {current:<10.2f} {suggested:<10.3f} {status}")
        
        # Find tests where tolerance can be reduced
        print("\n📉 Tolerance reduction opportunities:")
        reduced = False
        for r in results:
            current = current_tols.get(r['name'], 0.01)
            if r['suggested_tolerance'] < current * 0.5:
                reduction = (1 - r['suggested_tolerance']/current) * 100
                print(f"  {r['name']}: reduce from {current} to {r['suggested_tolerance']:.3f} (-{reduction:.0f}%)")
                reduced = True
        if not reduced:
            print("  None - all tolerances are appropriate")
                
        # Find tests that need higher tolerance
        print("\n📈 Tests needing tolerance adjustment:")
        increased = False
        for r in results:
            current = current_tols.get(r['name'], 0.01)
            if r['suggested_tolerance'] > current:
                increase = (r['suggested_tolerance']/current - 1) * 100
                print(f"  {r['name']}: increase from {current} to {r['suggested_tolerance']:.3f} (+{increase:.0f}%)")
                increased = True
        if not increased:
            print("  None - all tests pass with current tolerances")
    else:
        print("No results collected")

if __name__ == "__main__":
    main()
