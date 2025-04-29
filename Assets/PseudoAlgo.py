# algorithms.py

ALGORITHM_TEMPLATES = {
    # Searching Algorithms
    "linear_search": [
        {"type": "start", "text": "Start Linear Search"},
        {"type": "process", "text": "i = 0"},
        {"type": "decision", "text": "i < n?", "true_branch": 3, "false_branch": 5},
        {"type": "decision", "text": "array[i] == target?", "true_branch": 6, "false_branch": 4},
        {"type": "process", "text": "i = i + 1", "next": 2},
        {"type": "process", "text": "Return -1 (Target not found)"},
        {"type": "process", "text": "Return i (Target found at index i)"},
        {"type": "end", "text": "End Linear Search"}
    ],
    "binary_search": [
        {"type": "start", "text": "Start Binary Search"},
        {"type": "process", "text": "low = 0, high = n-1"},
        {"type": "decision", "text": "low <= high?", "true_branch": 3, "false_branch": 6},
        {"type": "process", "text": "mid = (low + high) // 2"},
        {"type": "decision", "text": "array[mid] == target?", "true_branch": 7, "false_branch": 5},
        {"type": "decision", "text": "array[mid] > target?", "true_branch": 8, "false_branch": 9},
        {"type": "process", "text": "Return -1 (Target not found)"},
        {"type": "process", "text": "Return mid (Target found at index mid)"},
        {"type": "process", "text": "high = mid - 1", "next": 2},
        {"type": "process", "text": "low = mid + 1", "next": 2},
        {"type": "end", "text": "End Binary Search"}
    ],
    "jump_search": [
        {"type": "start", "text": "Start Jump Search"},
        {"type": "process", "text": "step = int(n ** 0.5), i = 0"},
        {"type": "decision", "text": "i < n && array[i] < target?", "true_branch": 3, "false_branch": 4},
        {"type": "process", "text": "i = i + step", "next": 2},
        {"type": "process", "text": "Linear search from max(0, i-step) to min(i, n-1)"},
        {"type": "end", "text": "End Jump Search"}
    ],

    # Sorting Algorithms
    "bubble_sort": [
        {"type": "start", "text": "Start Bubble Sort"},
        {"type": "process", "text": "n = length of array"},
        {"type": "process", "text": "i = 0"},
        {"type": "decision", "text": "i < n-1?", "true_branch": 4, "false_branch": 10},
        {"type": "process", "text": "j = 0"},
        {"type": "decision", "text": "j < n-i-1?", "true_branch": 6, "false_branch": 9},
        {"type": "decision", "text": "array[j] > array[j+1]?", "true_branch": 7, "false_branch": 8},
        {"type": "process", "text": "Swap array[j] and array[j+1]"},
        {"type": "process", "text": "j = j + 1", "next": 5},
        {"type": "process", "text": "i = i + 1", "next": 3},
        {"type": "end", "text": "End Bubble Sort"}
    ],
    "selection_sort": [
        {"type": "start", "text": "Start Selection Sort"},
        {"type": "process", "text": "i = 0"},
        {"type": "decision", "text": "i < n-1?", "true_branch": 3, "false_branch": 8},
        {"type": "process", "text": "min_idx = i"},
        {"type": "process", "text": "j = i + 1"},
        {"type": "decision", "text": "j < n?", "true_branch": 6, "false_branch": 7},
        {"type": "decision", "text": "array[j] < array[min_idx]?", "true_branch": 7, "false_branch": 5},
        {"type": "process", "text": "min_idx = j", "next": 5},
        {"type": "process", "text": "Swap array[i] and array[min_idx]", "next": 2},
        {"type": "end", "text": "End Selection Sort"}
    ],
    "insertion_sort": [
        {"type": "start", "text": "Start Insertion Sort"},
        {"type": "process", "text": "i = 1"},
        {"type": "decision", "text": "i < n?", "true_branch": 3, "false_branch": 8},
        {"type": "process", "text": "key = array[i], j = i-1"},
        {"type": "decision", "text": "j >= 0 && array[j] > key?", "true_branch": 5, "false_branch": 7},
        {"type": "process", "text": "array[j+1] = array[j]"},
        {"type": "process", "text": "j = j - 1", "next": 4},
        {"type": "process", "text": "array[j+1] = key", "next": 2},
        {"type": "end", "text": "End Insertion Sort"}
    ],
    "merge_sort": [
        {"type": "start", "text": "Start Merge Sort"},
        {"type": "decision", "text": "low < high?", "true_branch": 2, "false_branch": 6},
        {"type": "process", "text": "mid = (low + high) // 2"},
        {"type": "process", "text": "MergeSort(low, mid)"},
        {"type": "process", "text": "MergeSort(mid+1, high)"},
        {"type": "process", "text": "Merge(low, mid, high)"},
        {"type": "end", "text": "End Merge Sort"}
    ],
    "quick_sort": [
        {"type": "start", "text": "Start Quick Sort"},
        {"type": "decision", "text": "low < high?", "true_branch": 2, "false_branch": 5},
        {"type": "process", "text": "pivot = Partition(low, high)"},
        {"type": "process", "text": "QuickSort(low, pivot-1)"},
        {"type": "process", "text": "QuickSort(pivot+1, high)", "next": 1},
        {"type": "end", "text": "End Quick Sort"}
    ],
    "heap_sort": [
        {"type": "start", "text": "Start Heap Sort"},
        {"type": "process", "text": "Build Max Heap from array"},
        {"type": "process", "text": "i = n-1"},
        {"type": "decision", "text": "i > 0?", "true_branch": 4, "false_branch": 6},
        {"type": "process", "text": "Swap array[0] with array[i]"},
        {"type": "process", "text": "Heapify(0, i)", "next": 3},
        {"type": "end", "text": "End Heap Sort"}
    ],

    # Graph Algorithms
    "depth_first_search": [
        {"type": "start", "text": "Start DFS"},
        {"type": "process", "text": "Mark current node as visited"},
        {"type": "process", "text": "Process current node"},
        {"type": "decision", "text": "Has unvisited neighbors?", "true_branch": 4, "false_branch": 5},
        {"type": "process", "text": "Recursively DFS on unvisited neighbor", "next": 1},
        {"type": "end", "text": "End DFS"}
    ],
    "breadth_first_search": [
        {"type": "start", "text": "Start BFS"},
        {"type": "process", "text": "Initialize queue"},
        {"type": "process", "text": "Enqueue start node"},
        {"type": "decision", "text": "Queue not empty?", "true_branch": 4, "false_branch": 9},
        {"type": "process", "text": "Dequeue node u"},
        {"type": "decision", "text": "u not visited?", "true_branch": 6, "false_branch": 3},
        {"type": "process", "text": "Mark u as visited"},
        {"type": "process", "text": "Process u"},
        {"type": "process", "text": "Enqueue unvisited neighbors of u", "next": 3},
        {"type": "end", "text": "End BFS"}
    ],
    "dijkstra": [
        {"type": "start", "text": "Start Dijkstra"},
        {"type": "process", "text": "Initialize distances from source to infinity, source to 0"},
        {"type": "process", "text": "Create priority queue with (distance, node)"},
        {"type": "decision", "text": "Queue not empty?", "true_branch": 4, "false_branch": 10},
        {"type": "process", "text": "Extract node u with minimum distance"},
        {"type": "decision", "text": "u not processed?", "true_branch": 6, "false_branch": 3},
        {"type": "process", "text": "Mark u as processed"},
        {"type": "process", "text": "For each neighbor v of u"},
        {"type": "decision", "text": "Distance via u < current distance to v?", "true_branch": 9, "false_branch": 3},
        {"type": "process", "text": "Update distance to v, enqueue v", "next": 3},
        {"type": "end", "text": "End Dijkstra"}
    ],
    "kruskal_mst": [
        {"type": "start", "text": "Start Kruskal"},
        {"type": "process", "text": "Sort all edges by weight"},
        {"type": "process", "text": "Initialize disjoint set for all vertices"},
        {"type": "decision", "text": "More edges to process?", "true_branch": 4, "false_branch": 7},
        {"type": "process", "text": "Get next edge (u, v)"},
        {"type": "decision", "text": "Union of u and v does not form cycle?", "true_branch": 6, "false_branch": 3},
        {"type": "process", "text": "Add edge (u, v) to MST", "next": 3},
        {"type": "end", "text": "End Kruskal"}
    ],
    "prim_mst": [
        {"type": "start", "text": "Start Prim"},
        {"type": "process", "text": "Initialize priority queue with (weight, vertex)"},
        {"type": "process", "text": "Select arbitrary start vertex"},
        {"type": "decision", "text": "Queue not empty?", "true_branch": 4, "false_branch": 8},
        {"type": "process", "text": "Extract minimum edge (u, v)"},
        {"type": "decision", "text": "v not visited?", "true_branch": 6, "false_branch": 3},
        {"type": "process", "text": "Mark v as visited, add edge to MST"},
        {"type": "process", "text": "Add edges from v to queue", "next": 3},
        {"type": "end", "text": "End Prim"}
    ],
    "topological_sort": [
        {"type": "start", "text": "Start Topological Sort"},
        {"type": "process", "text": "Initialize empty stack"},
        {"type": "decision", "text": "Unvisited nodes exist?", "true_branch": 3, "false_branch": 6},
        {"type": "process", "text": "DFS from an unvisited node"},
        {"type": "process", "text": "Push node to stack after DFS"},
        {"type": "process", "text": "Continue with next node", "next": 2},
        {"type": "end", "text": "End Topological Sort"}
    ],

    # Dynamic Programming
    "fibonacci": [
        {"type": "start", "text": "Start Fibonacci"},
        {"type": "process", "text": "dp[0] = 0, dp[1] = 1"},
        {"type": "process", "text": "i = 2"},
        {"type": "decision", "text": "i <= n?", "true_branch": 4, "false_branch": 6},
        {"type": "process", "text": "dp[i] = dp[i-1] + dp[i-2]"},
        {"type": "process", "text": "i = i + 1", "next": 3},
        {"type": "process", "text": "Return dp[n]"},
        {"type": "end", "text": "End Fibonacci"}
    ],
    "knapsack_01": [
        {"type": "start", "text": "Start 0/1 Knapsack"},
        {"type": "process", "text": "Initialize dp[n+1][W+1] to 0"},
        {"type": "process", "text": "i = 1"},
        {"type": "decision", "text": "i <= n?", "true_branch": 4, "false_branch": 10},
        {"type": "process", "text": "j = 0"},
        {"type": "decision", "text": "j <= W?", "true_branch": 6, "false_branch": 9},
        {"type": "decision", "text": "wt[i-1] <= j?", "true_branch": 7, "false_branch": 8},
        {"type": "process", "text": "dp[i][j] = max(val[i-1] + dp[i-1][j-wt[i-1]], dp[i-1][j])"},
        {"type": "process", "text": "dp[i][j] = dp[i-1][j]"},
        {"type": "process", "text": "j = j + 1", "next": 5},
        {"type": "process", "text": "i = i + 1", "next": 3},
        {"type": "end", "text": "End 0/1 Knapsack"}
    ],
    "longest_common_subsequence": [
        {"type": "start", "text": "Start LCS"},
        {"type": "process", "text": "Initialize dp[m+1][n+1] to 0"},
        {"type": "process", "text": "i = 1"},
        {"type": "decision", "text": "i <= m?", "true_branch": 4, "false_branch": 10},
        {"type": "process", "text": "j = 1"},
        {"type": "decision", "text": "j <= n?", "true_branch": 6, "false_branch": 9},
        {"type": "decision", "text": "str1[i-1] == str2[j-1]?", "true_branch": 7, "false_branch": 8},
        {"type": "process", "text": "dp[i][j] = dp[i-1][j-1] + 1"},
        {"type": "process", "text": "dp[i][j] = max(dp[i-1][j], dp[i][j-1])"},
        {"type": "process", "text": "j = j + 1", "next": 5},
        {"type": "process", "text": "i = i + 1", "next": 3},
        {"type": "end", "text": "End LCS"}
    ],

    # String Algorithms
    "kmp_string_matching": [
        {"type": "start", "text": "Start KMP"},
        {"type": "process", "text": "Compute LPS array for pattern"},
        {"type": "process", "text": "i = 0, j = 0"},
        {"type": "decision", "text": "i < n?", "true_branch": 4, "false_branch": 9},
        {"type": "decision", "text": "text[i] == pattern[j]?", "true_branch": 5, "false_branch": 6},
        {"type": "process", "text": "i = i + 1, j = j + 1"},
        {"type": "decision", "text": "j == m?", "true_branch": 7, "false_branch": 3},
        {"type": "process", "text": "Report match at (i-m)"},
        {"type": "process", "text": "j = lps[j-1] if j > 0 else 0", "next": 3},
        {"type": "end", "text": "End KMP"}
    ],
    "rabin_karp": [
        {"type": "start", "text": "Start Rabin-Karp"},
        {"type": "process", "text": "Compute pattern hash"},
        {"type": "process", "text": "Compute initial window hash"},
        {"type": "process", "text": "i = 0"},
        {"type": "decision", "text": "i <= n-m?", "true_branch": 5, "false_branch": 9},
        {"type": "decision", "text": "Hash match?", "true_branch": 6, "false_branch": 8},
        {"type": "process", "text": "Verify character match"},
        {"type": "process", "text": "Report match if verified"},
        {"type": "process", "text": "Slide window, update hash", "next": 4},
        {"type": "end", "text": "End Rabin-Karp"}
    ],

    # Mathematical Algorithms
    "euclidean_gcd": [
        {"type": "start", "text": "Start Euclidean GCD"},
        {"type": "decision", "text": "b == 0?", "true_branch": 3, "false_branch": 4},
        {"type": "process", "text": "Return a"},
        {"type": "process", "text": "Return GCD(b, a % b)", "next": 1},
        {"type": "end", "text": "End Euclidean GCD"}
    ],
    "sieve_of_eratosthenes": [
        {"type": "start", "text": "Start Sieve of Eratosthenes"},
        {"type": "process", "text": "Initialize isPrime[0..n] to true"},
        {"type": "process", "text": "isPrime[0] = isPrime[1] = false"},
        {"type": "process", "text": "i = 2"},
        {"type": "decision", "text": "i * i <= n?", "true_branch": 5, "false_branch": 9},
        {"type": "decision", "text": "isPrime[i]?", "true_branch": 6, "false_branch": 8},
        {"type": "process", "text": "j = i * i"},
        {"type": "decision", "text": "j <= n?", "true_branch": 8, "false_branch": 8},
        {"type": "process", "text": "isPrime[j] = false, j = j + i", "next": 7},
        {"type": "process", "text": "i = i + 1", "next": 4},
        {"type": "end", "text": "End Sieve of Eratosthenes"}
    ],

    # Additional Algorithms (to reach 50)
    "bellman_ford": [
        {"type": "start", "text": "Start Bellman-Ford"},
        {"type": "process", "text": "Initialize distances from source to infinity, source to 0"},
        {"type": "process", "text": "i = 0"},
        {"type": "decision", "text": "i < V-1?", "true_branch": 4, "false_branch": 6},
        {"type": "process", "text": "Relax all edges"},
        {"type": "process", "text": "i = i + 1", "next": 3},
        {"type": "decision", "text": "Negative cycle exists?", "true_branch": 8, "false_branch": 9},
        {"type": "process", "text": "Report negative cycle"},
        {"type": "end", "text": "End Bellman-Ford"}
    ],
    "floyd_warshall": [
        {"type": "start", "text": "Start Floyd-Warshall"},
        {"type": "process", "text": "Initialize distance matrix with infinity, diagonals to 0"},
        {"type": "process", "text": "k = 0"},
        {"type": "decision", "text": "k < V?", "true_branch": 4, "false_branch": 9},
        {"type": "process", "text": "i = 0"},
        {"type": "decision", "text": "i < V?", "true_branch": 6, "false_branch": 8},
        {"type": "process", "text": "Update dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])"},
        {"type": "process", "text": "i = i + 1", "next": 5},
        {"type": "process", "text": "k = k + 1", "next": 3},
        {"type": "end", "text": "End Floyd-Warshall"}
    ],
    "a_star": [
        {"type": "start", "text": "Start A*"},
        {"type": "process", "text": "Initialize open set with start node"},
        {"type": "decision", "text": "Open set not empty?", "true_branch": 3, "false_branch": 7},
        {"type": "process", "text": "Get node with minimum f-score"},
        {"type": "decision", "text": "Goal reached?", "true_branch": 6, "false_branch": 5},
        {"type": "process", "text": "Expand node, update neighbors", "next": 2},
        {"type": "process", "text": "Return path"},
        {"type": "end", "text": "End A*"}
    ],
    "huffman_coding": [
        {"type": "start", "text": "Start Huffman Coding"},
        {"type": "process", "text": "Build frequency table"},
        {"type": "process", "text": "Create priority queue of nodes"},
        {"type": "decision", "text": "Queue size > 1?", "true_branch": 4, "false_branch": 6},
        {"type": "process", "text": "Merge two nodes with minimum frequency"},
        {"type": "process", "text": "Generate codes from root", "next": 3},
        {"type": "end", "text": "End Huffman Coding"}
    ],
    "convex_hull_graham": [
        {"type": "start", "text": "Start Graham's Convex Hull"},
        {"type": "process", "text": "Find bottom-most point as pivot"},
        {"type": "process", "text": "Sort points by polar angle"},
        {"type": "decision", "text": "More points?", "true_branch": 4, "false_branch": 6},
        {"type": "process", "text": "Add point to hull, handle collinear points"},
        {"type": "process", "text": "Return hull", "next": 3},
        {"type": "end", "text": "End Graham's Convex Hull"}
    ],
    "convex_hull_jarvis": [
        {"type": "start", "text": "Start Jarvis March"},
        {"type": "process", "text": "Find leftmost point"},
        {"type": "decision", "text": "Hull not complete?", "true_branch": 3, "false_branch": 5},
        {"type": "process", "text": "Find next hull point using orientation"},
        {"type": "process", "text": "Return hull", "next": 2},
        {"type": "end", "text": "End Jarvis March"}
    ],
    "boyer_moore": [
        {"type": "start", "text": "Start Boyer-Moore"},
        {"type": "process", "text": "Precompute bad character and good suffix rules"},
        {"type": "process", "text": "i = 0"},
        {"type": "decision", "text": "i <= n-m?", "true_branch": 4, "false_branch": 6},
        {"type": "process", "text": "Align pattern, check mismatch"},
        {"type": "process", "text": "Shift pattern using rules", "next": 3},
        {"type": "end", "text": "End Boyer-Moore"}
    ],
    "trie_insert": [
        {"type": "start", "text": "Start Trie Insert"},
        {"type": "process", "text": "Start from root"},
        {"type": "process", "text": "For each character"},
        {"type": "decision", "text": "Node for char exists?", "true_branch": 5, "false_branch": 4},
        {"type": "process", "text": "Create new node"},
        {"type": "process", "text": "Move to next node", "next": 3},
        {"type": "process", "text": "Mark end of word"},
        {"type": "end", "text": "End Trie Insert"}
    ],
    "trie_search": [
        {"type": "start", "text": "Start Trie Search"},
        {"type": "process", "text": "Start from root"},
        {"type": "process", "text": "For each character"},
        {"type": "decision", "text": "Node for char exists?", "true_branch": 5, "false_branch": 6},
        {"type": "process", "text": "Move to next node", "next": 3},
        {"type": "process", "text": "Return false (not found)"},
        {"type": "decision", "text": "End marker?", "true_branch": 8, "false_branch": 6},
        {"type": "process", "text": "Return true (found)"},
        {"type": "end", "text": "End Trie Search"}
    ],
    "suffix_array": [
        {"type": "start", "text": "Start Suffix Array"},
        {"type": "process", "text": "Generate all suffixes"},
        {"type": "process", "text": "Sort suffixes lexicographically"},
        {"type": "end", "text": "End Suffix Array"}
    ],
    "manacher": [
        {"type": "start", "text": "Start Manacher's Algorithm"},
        {"type": "process", "text": "Transform string to handle odd/even lengths"},
        {"type": "process", "text": "Compute palindrome lengths"},
        {"type": "end", "text": "End Manacher's Algorithm"}
    ],
    "z_algorithm": [
        {"type": "start", "text": "Start Z Algorithm"},
        {"type": "process", "text": "Initialize Z array"},
        {"type": "process", "text": "Compute Z values using sliding window"},
        {"type": "end", "text": "End Z Algorithm"}
    ],
    "bipartite_matching": [
        {"type": "start", "text": "Start Bipartite Matching"},
        {"type": "process", "text": "Initialize matching and visited sets"},
        {"type": "decision", "text": "Unmatched nodes?", "true_branch": 3, "false_branch": 5},
        {"type": "process", "text": "Augment path using DFS/BFS"},
        {"type": "process", "text": "Update matching", "next": 2},
        {"type": "end", "text": "End Bipartite Matching"}
    ],
    "max_flow_ford_fulkerson": [
        {"type": "start", "text": "Start Ford-Fulkerson"},
        {"type": "process", "text": "Initialize residual graph"},
        {"type": "decision", "text": "Augmenting path exists?", "true_branch": 3, "false_branch": 5},
        {"type": "process", "text": "Find augmenting path using BFS/DFS"},
        {"type": "process", "text": "Update flow", "next": 2},
        {"type": "end", "text": "End Ford-Fulkerson"}
    ],
    "min_cut": [
        {"type": "start", "text": "Start Min Cut"},
        {"type": "process", "text": "Run max flow algorithm"},
        {"type": "process", "text": "Identify minimum cut set"},
        {"type": "end", "text": "End Min Cut"}
    ],
    "karger_min_cut": [
        {"type": "start", "text": "Start Karger's Min Cut"},
        {"type": "process", "text": "Contract random edges until 2 nodes remain"},
        {"type": "process", "text": "Count cut edges"},
        {"type": "end", "text": "End Karger's Min Cut"}
    ],
    "tarjan_scc": [
        {"type": "start", "text": "Start Tarjan's SCC"},
        {"type": "process", "text": "Initialize discovery and low-link values"},
        {"type": "process", "text": "DFS with stack for SCCs"},
        {"type": "end", "text": "End Tarjan's SCC"}
    ],
    "kosaraju_scc": [
        {"type": "start", "text": "Start Kosaraju's SCC"},
        {"type": "process", "text": "First DFS for finishing times"},
        {"type": "process", "text": "Transpose graph, second DFS for SCCs"},
        {"type": "end", "text": "End Kosaraju's SCC"}
    ],
    "articulation_points": [
        {"type": "start", "text": "Start Articulation Points"},
        {"type": "process", "text": "DFS with discovery and low values"},
        {"type": "process", "text": "Identify cut vertices"},
        {"type": "end", "text": "End Articulation Points"}
    ],
    "bridges": [
        {"type": "start", "text": "Start Bridges"},
        {"type": "process", "text": "DFS with discovery and low values"},
        {"type": "process", "text": "Identify bridges"},
        {"type": "end", "text": "End Bridges"}
    ],
    "euler_path": [
        {"type": "start", "text": "Start Euler Path"},
        {"type": "process", "text": "Check graph connectivity and degree"},
        {"type": "process", "text": "Find Eulerian path using Hierholzer's algorithm"},
        {"type": "end", "text": "End Euler Path"}
    ],
    "hamiltonian_cycle": [
        {"type": "start", "text": "Start Hamiltonian Cycle"},
        {"type": "process", "text": "Backtrack to find cycle"},
        {"type": "process", "text": "Check all vertices visited"},
        {"type": "end", "text": "End Hamiltonian Cycle"}
    ],
    "traveling_salesman_dp": [
        {"type": "start", "text": "Start TSP DP"},
        {"type": "process", "text": "Initialize dp with infinity"},
        {"type": "process", "text": "Use dynamic programming with bitmask"},
        {"type": "end", "text": "End TSP DP"}
    ],
    "matrix_chain_multiplication": [
        {"type": "start", "text": "Start Matrix Chain Multiplication"},
        {"type": "process", "text": "Initialize dp table"},
        {"type": "process", "text": "Fill dp using minimum cost"},
        {"type": "end", "text": "End Matrix Chain Multiplication"}
    ],
    "subset_sum": [
        {"type": "start", "text": "Start Subset Sum"},
        {"type": "process", "text": "Initialize dp array"},
        {"type": "process", "text": "Fill dp for possible sums"},
        {"type": "end", "text": "End Subset Sum"}
    ],
    "rod_cutting": [
        {"type": "start", "text": "Start Rod Cutting"},
        {"type": "process", "text": "Initialize dp array"},
        {"type": "process", "text": "Fill dp with max profit"},
        {"type": "end", "text": "End Rod Cutting"}
    ],
    "coin_change": [
        {"type": "start", "text": "Start Coin Change"},
        {"type": "process", "text": "Initialize dp array"},
        {"type": "process", "text": "Fill dp with minimum coins"},
        {"type": "end", "text": "End Coin Change"}
    ],
    "longest_increasing_subsequence": [
        {"type": "start", "text": "Start LIS"},
        {"type": "process", "text": "Initialize dp array"},
        {"type": "process", "text": "Fill dp with lengths"},
        {"type": "end", "text": "End LIS"}
    ],
    "edit_distance": [
        {"type": "start", "text": "Start Edit Distance"},
        {"type": "process", "text": "Initialize dp matrix"},
        {"type": "process", "text": "Fill dp with minimum edits"},
        {"type": "end", "text": "End Edit Distance"}
    ],
    "minimum_spanning_tree_boruvka": [
        {"type": "start", "text": "Start Boruvka's MST"},
        {"type": "process", "text": "Initialize disjoint sets"},
        {"type": "process", "text": "Iteratively find cheapest edges"},
        {"type": "end", "text": "End Boruvka's MST"}
    ]
}

def get_algorithm(name):
    """Return a copy of the specified algorithm template"""
    return ALGORITHM_TEMPLATES.get(name, []).copy()