# Data-Structure-Graphs

Graphs are mathematical structures that represent pairwise relationships between objects. A graph is a flow structure that represents the relationship between various objects. It can be visualized by using the following two basic components:
Nodes: These are the most important components in any graph. Nodes are entities whose relationships are expressed using edges. If a graph comprises 2 nodes and and an undirected edge between them, then it expresses a bi-directional relationship between the nodes and edge.
Edges: Edges are the components that are used to represent the relationships between various nodes in a graph. An edge between two nodes expresses a one-way or two-way relationship between the nodes.

=========
Types of graphs
- Undirected
- Directed
- Weighted
- Cyclic

=========
A tree is an undirected graph in which any two vertices are connected by only one path. A tree is an acyclic graph and has N - 1 edges where N is the number of vertices. Each node in a graph may have one or multiple parent nodes. However, in a tree, each node (except the root node) comprises exactly one parent node.

=========
Applications: relationship between components, transportation network, computer network

=========
The two most common ways of representing a graph is as follows:
1) Adjacency matrix: Space complexity is O(V2) constant time access (O(1))
2) Adjacency list: space complexity O(V + E) constant time access (O(Degree(V)))

=========
Two algorithms for traversing the graph: DFS and BFS

======
Which data structures are used for BFS and DFS of a graph?
Queue is used for BFS. Stack is used for DFS. DFS can also be implemented using recursion (Note that recursion also uses function call stack).

=========
DFS algorithm works in a manner similar to preorder traversal of trees. Internally this algo uses stack. We have a adjacency matrix to store the link between the vertices. An array of vertex and each vertex has the variable label and visited. 
public class DFS {
    private int count;
    private Stack stack;
    private int ajMatrix[][];
    private int numberOfVertex;
    private Vertex vertexList[];
    public void initializeGraph(int vertex) {
        count = 0;
        stack = new Stack();
        numberOfVertex = vertex;
        ajMatrix = new int[numberOfVertex][numberOfVertex];
        vertexList = new Vertex[numberOfVertex];
        for (int i = 0; i < numberOfVertex; i++) {
            for (int j = 0; j < numberOfVertex; j++) {
                ajMatrix[i][j] = 0;
            }
        }
    }
    // add the vertex
    public void addVertex(char label) {
        Vertex vertex = new Vertex();
        vertex.setLabel(label);
        vertexList[count] = vertex;
        count++;
    }
    public void addEdge(int start, int end) {
        ajMatrix[start][end] = 1;
        ajMatrix[end][start] = 1;
    }
    public void displayVertex(int i) {
        System.out.println(vertexList[i].getLabel());
    }
    public void dfsUsingStack() {
        displayVertex(0);
        stack.push(0);
        vertexList[0].setVisited(true);
        int vertexNumber;
        while (!stack.empty()) {
            vertexNumber = (int) stack.peek();
            int adjUnvisitedVertex = getAdjUnvisitedVertex(vertexNumber);
            if (adjUnvisitedVertex != -1) {
                stack.push(adjUnvisitedVertex);
                vertexList[adjUnvisitedVertex].setVisited(true);
                displayVertex(adjUnvisitedVertex);
            } else {
                stack.pop();
            }
        }
    }
    public int getAdjUnvisitedVertex(int i) {
        for (int j = 0; j < numberOfVertex; j++) {
            if (ajMatrix[i][j] == 1 && !vertexList[j].isVisited()) {
                return j;
            }
        }
        return -1;
    }
}
Applications: Topological sorting, finding connected components, solving puzzles such as mazes.

==========
BFS works similar to level order traversing. We use queues here.
public class BFS {
    private int count;
    private Queue queue;
    private int adjMatrix[][];
    private int numberOfVertex;
    private Vertex vertexList[];
    public void initializeGraph(int numberOfVertex) {
        count = 0;
        queue = new Queue();
        this.numberOfVertex = numberOfVertex;
        vertexList = new Vertex[numberOfVertex];
        adjMatrix = new int[numberOfVertex][numberOfVertex];
    }
    public void addVertex(char label) {
        Vertex vertex = new Vertex();
        vertex.setLabel(label);
        vertexList[count] = vertex;
        count++;
    }
    public void addEdge(int start, int end) {
        adjMatrix[start][end] = 1;
        adjMatrix[end][start] = 1;
    }
    public void displayVertex(int vertexNumber) {
        System.out.println(vertexList[vertexNumber].getLabel());
    }
    public void bfsUsingQueue() {
        queue.enqueue(0);
        displayVertex(0);
        vertexList[0].setVisited(true);
        int currentVertex, adjUnvisitedVertex;
        while (!queue.empty()) {
            currentVertex = (int) queue.dequeue();
            while (true) {
                adjUnvisitedVertex = getAdjUnvisitedNode(currentVertex);
                if (adjUnvisitedVertex != -1) {
                    queue.enqueue(adjUnvisitedVertex);
                    vertexList[adjUnvisitedVertex].setVisited(true);
                    displayVertex(adjUnvisitedVertex);
                } else {
                    break;
                }
            }
        }
    }
    public int getAdjUnvisitedNode(int i) {
        for (int j = 0; j < numberOfVertex; j++) {
            if (adjMatrix[i][j] == 1 && !vertexList[j].isVisited()) {
                return j;
            }
        }
        return -1;
    }
}
Applications: finding connected components in a graph, shortest path

=========
Which one is preferred when
- Spanning forest, connected components, paths, cycles: BFS, DFS
- Shortest paths: BFS
- Minimal use of memory: DFS as BFS has to have the connected vertices added to the queue taking up more space.

=========
Topological sort: is an ordering of vertices in a directed acyclic graph in which each node comes before all nodes to which it has outgoing edges.
        A           B
          \       /  \   
              C       D
               \      /   
                E    /
                 \  /
                   F
                   /
                  G
So topological sort for above graph can be: ABCEDFG, BDACEFG. We need to have a set has all visited vertices and stack has all the vertices in topological sorted order. Let's start from any node. We put E in the visited set and explore the children of E. If there is no children or all the children of that vertex is explored then that vertex is put into the stack.
public Stack<Vertex> topSort(Graph graph){
    Stack<Vertex> stack = new Stack();
    Set<Vertex> visited = new HashSet();
    for(Vertex v: graph.getAllVertex){
        if(visited.contains(v)){
            continue;
        }
        topSortUtil(v, stack, visited);
    }
}
public topSortUtil(Vertex vertex, Stack<Vertex> stack, Set<Vertex> visited){
    visited.add(v);
    for(Vertex v: graph.getAdjacentVertices){
        if(visited.contains(v)){
            continue;
        }
        topSortUtil(v, stack, visited);
    }   
    stack.push(vertex);
}
https://www.youtube.com/watch?v=ddTC4Zovtbc

=========
Disjoint sets (using union rank and path compression) is a data structure that supports three operations:
1) Makeset(X): Creates a new set containing a single element x
2) Union(X,Y): Creates a new set containing the elements X and Y with their union and deletes the sets containing the elements X and Y
3) Findset(X): Returns the name of the set containing the element X. It
Application: Network connectivity, image processing, LCA, Kruskal algo, used to find cycle in an undirected graph
https://www.youtube.com/watch?v=ID00PMy0-vE

=========
Implementation:
private Map<Long, Node> map = new HashMap();
Node of tree for Disjoint sets
class Node{
    int rank; // approximate depth of the tree
    int data;
    Node parent;
}
Union by rank says make the higher rank the parent and lower rank the child. Find operation traverses list of nodes on the way to the root. We can make later FIND operations efficient by making each of these vertices point directly to the root. This process is called path compression.

// creates the set with one element
public void makeSet(long data){
    Node node = new Node();
    node.data = data;
    node.parent = node;
    node.rank = 0;
    map.put(data, node);
}
// combines two sets together to one. Does union by rank
public void union(long data1, long data2){
    Node node1 = map.get(data1);
    Node node2 = map.get(data2);

    Node parent1 = findSet(node1);
    Node parent2 = findSet(node2);

    // if the parent are same do nothing
    if(parent1.data == parent2.data){
        return;
    }

    // else whoever's rank is higher becomes parent of other
    if(parent1.rank >= parent2.rank){
        // increment rank only if both sets have same rank
        parent1.rank = (parent1.rank == parent2.rank) ? parent1.rank + 1 : parent1.rank;
    }else{
        parent1.parent= parent2;
    }
}

// find the representative recursively and does path compression as well
public Node findSet(Node node){
    Node parent = node.parent;
    if(parent == node){
        return parent;
    }
    node.parent = findSet(node.parent);
    return node.parent;
}

// find the representative of this set
public long findSet(long data){
    return findSet(map.get(data)).data;
}
https://www.youtube.com/watch?v=ID00PMy0-vE

=========
Spanning tree of a undirected graph is a subgraph that contains all the vertices and connected to each other and there are n-1 edges in the subgraph where n is total number of vertices. Basically it is a tree. Minimum spanning tree is a spanning tree such that sum of the edges of the tree is minimum. A spanning tree does not have cycles and it cannot be disconnected. A complete undirected graph can have maximum n^(n-2) number of spanning trees, where n is the number of vertices.

=========
Kruskal's algorithm for Minimum spanning tree: Firstly we sort the edges in non-decreasing order. We create as many disjoint sets as many number of vertices. Now starting from the first edge we join the two sets. If in the edge XY, X is present in different set and Y in different set then that Edge will be included in the result and both the sets will be joined. If in edge XY, both the X and Y are present in the same set than edge XY is ignored. Finally the values which are present in the result will constitute the minimum spanning tree. 

public List<Edge> getMST(Graph graph){
    List<Edge> allEdges = graph.allEdges();
    EdgeComparater comp = new EdgeComparater();

    // sort the edges in non-decreasing order
    Collections.sort(allEdges, comp);
    DisjointSet disjoinSet = new DisjoinSet();

    // create as many disjoint sets as the total vertices
    for(Vertex vertex : graph.allVertex()){
        disjoinSet.makeSet(vertex.getId());
    }

    List<Edge> resultEdges = new ArrayList();
    for(Edge edge: allEdges){
        long root1 = disjoinSet.findSet(edge.getVertex1().getId());
        long root2 = disjoinSet.findSet(edge.getVertex2().getId());

        // check if the vertices are in the same set or different set. if vertices are in same set then ignore the edge
        if(root1 == root2){
            continue;
        }else{
            // if vertices are in different set then add the edge to result and union these two sets into one
            resultEdges.add(edge);
            DisjoinSet.union(edge.getVertex1().getId(), edge.getVertext2().getId());
        }
    }
    return resultEdges;
}
https://www.youtube.com/watch?v=fAuF0EuZVCk

=========
Prim's Algorithm Minimum Spanning Tree Graph Algorithm: We need to have minimum binary heap or priority queue and a map to take care the positioning of vertices in array after heapify. First A is set to value 0 and rest to infinity. We put all the vertices in the queue or binary heap. Extract min from queue. And check vertices that are attached to Vertex A. Put the distances between AD and AB to D and B respectively. And then extract min again. We also maintain the V and E from where the min value is coming.
public List<Edge> primMST(Graph graph){
    // binary heap + map data structure
    Heap<Vertex> minHeap = new Heap();

    // map of vertex to edge which gave minimum weight to this vertex;
    Map<Vertex, Edge> vertexToEdge = new HashMap();

    // stores final result
    List<Edge> result = new ArrayList();

    // insert all vertices with infinite value initially
    for(Vertex v: graph.getAllVertex()){
        minHeap.add(Integer.MAX_VALUE, v);
    }

    // start from any random vertex
    Vertex startVertex = graph.getAllVertex().iterator().next();

    // for the start vertex decrease the value in heap to 0
    minHeap.decrease(startVertex, 0);

    // iterate till heap has elements in it
    while(!minHeap.isEmpty){
        // extract min value vertex from heap
        Vertex current = minHeap.extractMin();

        // get the corresponding edge for this vertex if present and add it to final result
        // this edge won't be present for first vertex
        Edge spanningTreeEdge = vertexToEdge(current);
        if(spanningTreeEdge!= null){
            result.add(spanningTreeEdge);
        }

        // iterate through all the adjacent vertices
        for(Edge edge: current.getEdges()){
            Vertex adjacent = getVertexForEdge(current, edge);

            // check if the adjacent vertex exist in heap and weight attached with this vertex is greater than this edge weight
            if(minHeap.containsData(adjacent) && minHeap.getWeight(adjacent) > edge.getWeight()){
                // decrease the value of adjacent vertex to this edge weight
                minHeap.decrease(adjacent, edge.getWeight());

                // add vertex-edge mapping in the graph
                vertexToEdge.put(adjacent, edge);
            }
        }
    }
    return result;
}

https://www.youtube.com/watch?v=oP2-8ysT3QQ

=========
Shortest path in unweighted graph: It is special case of weighted shortest-path problem, with all the edges weight of 1. Algorithm is similar to BFS. Here we use the distance table. We keep the shortest distance from the vertex in the distance table. Just the difference is in place of priority queue we can use the normal queue as the edge weights are 1 only.

=========
Dijkstra: Like unweighted graph, here also we use the distance table. We keep the shortest distance from the vertex in the distance table. It is greedy algorithm. It picks the next closest vertex to the source. Uses priority queue to store the unvisited vertices by distance from source. Does not work with negative weights.
public class Dijkstra {
    private int count;
    private int adjMatrix[][];
    private int numberOfVertex;
    private Vertex vertexList[];
    private PriorityQueue<Vertex> queue;
    public void initializeGraph(int numberOfVertex) {
        count = 0;
        this.numberOfVertex = numberOfVertex;
        vertexList = new Vertex[numberOfVertex];
        adjMatrix = new int[numberOfVertex][numberOfVertex];
        queue = new PriorityQueue<Vertex>(5, new Comparator<Vertex>() {
            @Override
            public int compare(Vertex o1, Vertex o2) {
                return o1.getMinDistance() - o2.getMinDistance();
            }
        });
    }
    public void addVertex(char label) {
        Vertex vertex = new Vertex();
        vertex.setLabel(label);
        vertex.setIndex(count);
        vertex.setMinDistance(Integer.MAX_VALUE);
        vertex.setParent(Character.MAX_VALUE);
        vertexList[count] = vertex;
        count++;
    }
    public void addEdge(int source, int destination, int weight) {
        adjMatrix[source][destination] = weight;
    }
    public void dijkstra() {
        Vertex currentVertex;
        vertexList[0].setMinDistance(0);
        queue.add(vertexList[0]);
        queue.add(vertexList[1]);
        queue.add(vertexList[2]);
        queue.add(vertexList[3]);
        queue.add(vertexList[4]);
        while (!queue.isEmpty()) {
            currentVertex = queue.remove();
            vertexList[currentVertex].visited = true
            int currentVertexIndex = currentVertex.getIndex();
            int neighbourVertexIndex = -1;

            while (true) {
                neighbourVertexIndex = getAdjUnvisitedVertex(currentVertex.getIndex(), neighbourVertexIndex);

                // check if there is no neighbor that is to be visited
                if (neighbourVertexIndex != -1) {

                    // check the distance between two
                    if (vertexList[currentVertexIndex].getMinDistance()
                            + adjMatrix[currentVertexIndex][neighbourVertexIndex] < vertexList[neighbourVertexIndex].getMinDistance()) {

                        // remove the vertex from the queue
                        queue.remove(vertexList[neighbourVertexIndex]);

                        // set the min distance
                        vertexList[neighbourVertexIndex].setMinDistance(vertexList[currentVertexIndex].getMinDistance()
                                + adjMatrix[currentVertexIndex][neighbourVertexIndex]);

                        // set the parent
                        vertexList[neighbourVertexIndex].setParent(currentVertex.getLabel());

                        // add the vertex again
                        queue.add(vertexList[neighbourVertexIndex]);
                    }
                } else {
                    break;
                }
            }
        }
        // print distance and parent
        for (int i = 0; i < numberOfVertex; i++) {
            Vertex vertex = vertexList[i];
            System.out.println(vertex.getLabel() + "-> Distance: " + vertex.getMinDistance() + " ,Parent: " + vertex.getParent());
        }
    }
    public int getAdjUnvisitedVertex(int currentVertexIndex, int startSearchVertexIndex) {
        for (int j = startSearchVertexIndex + 1; j < numberOfVertex; j++) {
            if (adjMatrix[currentVertexIndex][j] > 0 && vertexList[j].isVisited == false){
                return j;
            }
        }
        return -1;
    }
}
public class Vertex {
    private int index;
    private char label;
    private char parent;
    private int minDistance;
    private boolean isVisited;
}
Disadvantage: It does blind search thereby wasting time and necessary resources. Cannot handle negative edges.
https://www.youtube.com/watch?v=lAXZGERcDf4

========
Detecting the cycle in graph. If we use DFS, then if there is cycle the same A vertex will again be hit means the already visited vertex will be visited again.
