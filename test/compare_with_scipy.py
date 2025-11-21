#!/usr/bin/env python3
"""
Compare mathjs logm implementation with scipy.linalg.logm
Tests the same matrices and reports accuracy differences
"""

import numpy as np
import scipy.linalg
import subprocess
import json
import sys

# Test matrices matching those in logm.test.js
test_matrices = {
    "2x2_diagonal": [[np.e, 0], [0, np.e**2]],
    "2x2_upper_triangular": [[1, 2], [0, 1]],
    "2x2_rotation": [[np.cos(np.pi/4), -np.sin(np.pi/4)], 
                     [np.sin(np.pi/4), np.cos(np.pi/4)]],
    "3x3_upper_triangular": [[1, 2, 3], [0, 1, 4], [0, 0, 1]],
    "2x2_general": [[1, 2], [3, 4]],
    "5x5_mixed_blocks": [
        [1, 2, 0, 0, 0],
        [-2, 1, 1, 0, 0],
        [0, 0, 3, 0, 0],
        [0, 0, 0, 2, 1],
        [0, 0, 0, -1, 2]
    ],
}

def call_mathjs_logm(matrix):
    """Call mathjs logm via node.js"""
    matrix_json = json.dumps(matrix)
    code = f"""
    const math = require('./lib/cjs/index.js');
    const A = {matrix_json};
    try {{
        const result = math.logm(A);
        // Convert result to array format, handling complex numbers
        const arr = result.toArray ? result.toArray() : result;
        // Convert complex numbers to {{re, im}} format
        const serializable = arr.map(row => 
            row.map(val => {{
                if (typeof val === 'object' && val !== null && ('re' in val || 'im' in val)) {{
                    return {{re: val.re || 0, im: val.im || 0}};
                }}
                return {{re: val, im: 0}};
            }})
        );
        console.log(JSON.stringify(serializable));
    }} catch(e) {{
        console.error('Error:', e.message);
        process.exit(1);
    }}
    """
    
    result = subprocess.run(
        ['node', '-e', code],
        cwd='/home/runner/work/mathjs/mathjs',
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"mathjs error: {result.stderr}")
    
    parsed = json.loads(result.stdout)
    # Convert from {{re, im}} format to numpy complex
    return np.array([[complex(cell['re'], cell['im']) for cell in row] for row in parsed])

def compare_logm(name, A):
    """Compare mathjs and scipy logm results"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"Matrix shape: {np.array(A).shape}")
    
    # Compute with scipy
    scipy_result = scipy.linalg.logm(A)
    
    # Compute with mathjs
    try:
        mathjs_result = call_mathjs_logm(A)
    except Exception as e:
        print(f"  mathjs FAILED: {e}")
        return None
    
    # Compare results
    diff = np.abs(scipy_result - mathjs_result)
    max_diff = np.max(diff)
    rel_error = max_diff / (np.max(np.abs(scipy_result)) + 1e-16)
    
    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  Max relative error: {rel_error:.2e}")
    
    # Test roundtrip: expm(logm(A)) ≈ A
    scipy_roundtrip = scipy.linalg.expm(scipy_result)
    mathjs_roundtrip = scipy.linalg.expm(mathjs_result)
    
    scipy_roundtrip_error = np.max(np.abs(scipy_roundtrip - A)) / (np.max(np.abs(A)) + 1e-16)
    mathjs_roundtrip_error = np.max(np.abs(mathjs_roundtrip - A)) / (np.max(np.abs(A)) + 1e-16)
    
    print(f"  Scipy roundtrip error: {scipy_roundtrip_error:.2e}")
    print(f"  Mathjs roundtrip error: {mathjs_roundtrip_error:.2e}")
    
    # Determine if 150% tolerance is reasonable
    if mathjs_roundtrip_error > 1.5:
        print(f"  ⚠️  EXCEEDS 150% tolerance! ({mathjs_roundtrip_error*100:.1f}%)")
    elif mathjs_roundtrip_error > 0.5:
        print(f"  ⚠️  High error but within 150% ({mathjs_roundtrip_error*100:.1f}%)")
    elif mathjs_roundtrip_error > 0.1:
        print(f"  ⚠️  Moderate error ({mathjs_roundtrip_error*100:.1f}%)")
    else:
        print(f"  ✓ Good accuracy ({mathjs_roundtrip_error*100:.3f}%)")
    
    return {
        'name': name,
        'max_diff': float(max_diff),
        'rel_error': float(rel_error),
        'scipy_roundtrip': float(scipy_roundtrip_error),
        'mathjs_roundtrip': float(mathjs_roundtrip_error)
    }

def main():
    print("Comparing mathjs logm with scipy.linalg.logm")
    print("="*60)
    
    results = []
    for name, A in test_matrices.items():
        result = compare_logm(name, A)
        if result:
            results.append(result)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Test':<25} {'Max Diff':<12} {'Roundtrip%':<12} {'Status'}")
    print("-"*60)
    
    for r in results:
        status = "✓" if r['mathjs_roundtrip'] <= 0.1 else "⚠️"
        print(f"{r['name']:<25} {r['max_diff']:<12.2e} {r['mathjs_roundtrip']*100:<12.3f} {status}")
    
    print(f"\nMax roundtrip error: {max(r['mathjs_roundtrip'] for r in results)*100:.3f}%")
    
    # Check if 150% is needed
    max_error = max(r['mathjs_roundtrip'] for r in results)
    if max_error > 1.5:
        print(f"⚠️  Some tests exceed 150% tolerance!")
        sys.exit(1)
    elif max_error > 0.5:
        print(f"⚠️  Tolerance of {max_error*100:.1f}% needed (>50%)")
    elif max_error > 0.1:
        print(f"✓ Tolerance of {max_error*100:.1f}% needed (>10%)")
    else:
        print(f"✓ All tests have <10% error, 150% tolerance is very conservative")

if __name__ == "__main__":
    main()
