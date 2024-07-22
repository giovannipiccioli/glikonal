#include <vector>
#include <cmath>
#include <limits>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <array>
#include <queue>
#include <algorithm>
#include <chrono>


using Position = std::array<int, 2>;
using Edge = std::pair<Position, Position>;

//#include "grrp_funcs.h"


 //compile with g++ glikonal_grrp.cpp -o glikonal_grrp_out -std=c++17 -O2
 //run with ./glikonal_grrp_out -h 1 -hz 20 -H 100 -W 100 -Z 10 -fwind ./data/wind.txt -felev ./data/elevation.txt  -finit ./data/init_values_chars.txt -out_alt ./output/alt.txt -out_char ./output/char.txt -v
// Runs the Glikonal-G algorithm

/*
Takes as input:
the grid spacing h
the vertical spacing hz
the number of grid points in the x direction, W
the number of grid points in the y direction, H
the number of wind layers in the Z direction, Z
the name of the file containing the wind vector field. The aircraft is assumed to fly at fixed airspeed, the strength of the wind should represent the ration between the wind speed and the airspeed of the aircraft. The wind intensity should always be smaller than one.
the name of the file containing the elevation data
the name of the file containing the initial values of the function z_0-U_G, and the characteristics
the name of the file to write the function z_0-U_G
the name of the file where to write the characteristics


to change the glide ratio, change the value of the variable base_glide_ratio in the function velocity_function. base_glide_ratio is the glide ratio in absence of wind. From this the glide ratio in presence of wind is computed assuming the airspeed is fixed to 1.

*/ 

double** alloc_matrix(int H, int W){
    double** matrix = (double**)malloc(H * sizeof(double*));
    for (int i = 0; i < H; ++i) {
        matrix[i] = (double*)malloc(W * sizeof(double));
    }
    return matrix;
}

void free_matrix(double** matrix, int H){
    for (int i = 0; i < H; ++i) {
        free(matrix[i]);
    }
    free(matrix);
}

double*** alloc_tensor3(int H, int W, int Z){
    double*** tensor = (double***)malloc(H * sizeof(double**));
    for (int i = 0; i < H; ++i) {
        tensor[i] = (double**)malloc(W * sizeof(double*));
        for (int j = 0; j < W; ++j) {
            tensor[i][j] = (double*)malloc(Z * sizeof(double));
        }
    }
    return tensor;
}

void free_tensor3(double*** tensor, int H, int W){
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            free(tensor[i][j]);
        }
        free(tensor[i]);
    }
    free(tensor);
}

double**** alloc_tensor4(int H, int W, int Z, int T){
    double**** tensor = (double****)malloc(H * sizeof(double***));
    for (int i = 0; i < H; ++i) {
        tensor[i] = (double***)malloc(W * sizeof(double**));
        for (int j = 0; j < W; ++j) {
            tensor[i][j] = (double**)malloc(Z * sizeof(double*));
            for (int k = 0; k < Z; ++k) {
                tensor[i][j][k] = (double*)malloc(T * sizeof(double));
            }
        }
    }
    return tensor;
}
    
void free_tensor4(double**** tensor, int H, int W, int Z){
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            for (int k = 0; k < Z; ++k) {
                free(tensor[i][j][k]);
            }
            free(tensor[i][j]);
        }
        free(tensor[i]);
    }
    free(tensor);
}

bool** alloc_bool(int H, int W){
   bool** boolArray = new bool*[H];
    for (int i = 0; i < H; ++i) {
        boolArray[i] = new bool[W];
    }

    // Initialize the 2D array to false
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            boolArray[i][j] = false;
        }
    }
    return boolArray;
}

void free_bool(bool** boolArray, int H){
    for (int i = 0; i < H; ++i) {
        delete[] boolArray[i];
    }
    delete[] boolArray;
}

void showUsage() {
    std::cout << "Usage: glikonal_grrp -h <h> -hz <hz> -H <H> -W <W> -fwind <wind file> -felev <elevation file> -finit <initialization file> -out_alt <altitude file> -out_char <characteristics file>" << std::endl;
}

// Function that returns the list of neighbors of a given position in a triangulated grid graph.
// The triangulation is obtained by connecting node (i,j) with (i+1,j+1), in addition to the 4 adjacent nodes.

std::vector<Position> find_neighbors(Position pos, int H, int W) {
    int y = pos[0];
    int x = pos[1];
    std::vector<Position> neigh_list = {
        {y + 1, x},     // (y+1, x)
        {y, x + 1},     // (y, x+1)
        {y - 1, x},     // (y-1, x)
        {y, x - 1},     // (y, x-1)
        {y + 1, x + 1}, // (y+1, x+1)
        {y - 1, x - 1}  // (y-1, x-1)
    };

    // Remove neighbors based on boundaries
    if (y == 0) {
        neigh_list.erase(std::remove(neigh_list.begin(), neigh_list.end(), Position{y - 1, x}), neigh_list.end());
        neigh_list.erase(std::remove(neigh_list.begin(), neigh_list.end(), Position{y - 1, x - 1}), neigh_list.end());
    } else if (y == H - 1) {
        neigh_list.erase(std::remove(neigh_list.begin(), neigh_list.end(), Position{y + 1, x}), neigh_list.end());
        neigh_list.erase(std::remove(neigh_list.begin(), neigh_list.end(), Position{y + 1, x + 1}), neigh_list.end());
    }

    if (x == 0) {
        neigh_list.erase(std::remove(neigh_list.begin(), neigh_list.end(), Position{y, x - 1}), neigh_list.end());
        if (y > 0) {
            neigh_list.erase(std::remove(neigh_list.begin(), neigh_list.end(), Position{y - 1, x - 1}), neigh_list.end());
        }
    } else if (x == W - 1) {
        neigh_list.erase(std::remove(neigh_list.begin(), neigh_list.end(), Position{y, x + 1}), neigh_list.end());
        if (y < H - 1) {
            neigh_list.erase(std::remove(neigh_list.begin(), neigh_list.end(), Position{y + 1, x + 1}), neigh_list.end());
        }
    }

    return neigh_list;
}

bool is_in_grid(Position pos, int H, int W) {
    int y = pos[0];
    int x = pos[1];
    
    if (y < 0 || y >= H || x < 0 || x >= W) {
        return false;
    } else {
        return true;
    }
}

bool is_edge(Position pos1, Position pos2) {
    // Extract coordinates from tuples
    int y1 = pos1[0];
    int x1 = pos1[1];
    int y2 = pos2[0];
    int x2 = pos2[1];
    
    // Check conditions for edge connection
    if (std::abs(y1 - y2) > 1 || std::abs(x1 - x2) > 1 || (y1 - y2) * (x1 - x2) < 0 || pos1 == pos2) {
        return false;
    } else {
        return true;
    }
}


// Main function implementation
std::vector<Edge> compute_near_front(const Position& pos, 
                                     const bool* const * mask_accepted_front, 
                                     double max_dist_edge, 
                                     double hy, 
                                     double hx, 
                                     int H, 
                                     int W) {
    double triang_diameter = std::sqrt(hx * hx + hy * hy);
    double max_dist_node = max_dist_edge * std::sqrt(1 + (triang_diameter / max_dist_edge) * (triang_diameter / max_dist_edge));
    int max_graph_dist_node_x = static_cast<int>(std::ceil(max_dist_node / hx));
    int max_graph_dist_node_y = static_cast<int>(std::ceil(max_dist_node / hy));

    std::vector<Position> near_front_nodes;

    // Enumerate all points in the square of size 2 * max_dist_node
    for (int dy = -max_graph_dist_node_y; dy <= max_graph_dist_node_y; ++dy) {
        for (int dx = -max_graph_dist_node_x; dx <= max_graph_dist_node_x; ++dx) {
            Position neighbor = {pos[0] + dy, pos[1] + dx};
            if (is_in_grid(neighbor, H, W) && mask_accepted_front[neighbor[0]][neighbor[1]] 
                && (hy * dy) * (hy * dy) + (hx * dx) * (hx * dx) <= max_dist_node * max_dist_node) {
                near_front_nodes.push_back(neighbor);
            }
        }
    }

    // Include relevant nodes that are neighbors of the near_front_nodes
    std::vector<Position> new_near_front_nodes;
    for (const auto& pos_nf : near_front_nodes) {
        for (const auto& pos2 : find_neighbors(pos_nf, H, W)) {
            if (mask_accepted_front[pos2[0]][pos2[1]] 
                && std::find(near_front_nodes.begin(), near_front_nodes.end(), pos2) == near_front_nodes.end() 
                && std::find(new_near_front_nodes.begin(), new_near_front_nodes.end(), pos2) == new_near_front_nodes.end()) {
                new_near_front_nodes.push_back(pos2);
            }
        }
    }
    near_front_nodes.insert(near_front_nodes.end(), new_near_front_nodes.begin(), new_near_front_nodes.end());

    // Generate the list of near front edges
    std::vector<Edge> near_front_edges;
    for (size_t i = 0; i < near_front_nodes.size(); ++i) {
        for (size_t j = i + 1; j < near_front_nodes.size(); ++j) {
            if (is_edge(near_front_nodes[i], near_front_nodes[j])) {
                near_front_edges.push_back({near_front_nodes[i], near_front_nodes[j]});
            }
        }
    }

    return near_front_edges;
}

double velocity_function(const std::array<double, 2>& a, const std::array<double, 2>& wind) {
    const double base_glide_ratio = 1.0;
    double k = std::sqrt(wind[0]*wind[0]+wind[1]*wind[1]);  // Ratio between the wind speed and the airspeed of the aircraft

    double a_dot_wind=a[0]*wind[0] + a[1]*wind[1];
    // Calculate the glide ratio considering the wind effect
    return base_glide_ratio * (std::sqrt(1 - k * k + k * k * a_dot_wind * a_dot_wind) + k * a_dot_wind);
}

double V_simplex_zeta(double z, 
                      const Position& pos1, 
                      const Position& pos2, 
                      const Position& pos, 
                      const double U1, 
                      const double U2, 
                      double  (*velocity_func)(const std::array<double, 2>&, const std::array<double, 2>&), 
                      const std::array<double, 2>& args_velocity_func, 
                      double hy, 
                      double hx) {
    std::array<double, 2> a_unnorm = {hy*(pos[0] - z * pos1[0] - (1 - z) * pos2[0]), hx*(pos[1] - z * pos1[1] - (1 - z) * pos2[1])};
    
    double tau = std::sqrt(a_unnorm[0]*a_unnorm[0] + a_unnorm[1]*a_unnorm[1]);
    
    std::array<double, 2> direction = {a_unnorm[0] / tau, a_unnorm[1] / tau};
    double velocity =  velocity_func(direction, args_velocity_func);
    
    return tau / velocity + z * U1 + (1 - z) * U2;
    
}

// Function to compute the upwind update of the velocity field at a given position
std::pair<double, std::array<double, 2>> upwind_simplex_update(
    const Position& pos1,
    const Position& pos2,
    const Position& pos,
    double U1,
    double U2,
    double  (*velocity_func)(const std::array<double, 2>&, const std::array<double, 2>&),
    const std::array<double, 2>& args_velocity_func,
    double hy,
    double hx
) {
    double min_V=std::numeric_limits<double>::infinity();
    double z_min=0.0;
    for (int i = 0; i <= 10; ++i) {
        double z = static_cast<double>(i) / 10.0; 
        double V = V_simplex_zeta(z, pos1, pos2, pos, U1, U2, velocity_func, args_velocity_func, hy, hx);
        if (V < min_V) {
            min_V = V;
            z_min = z;
        }
    }

    std::array<double, 2> pos_zmin = {z_min * pos1[0] +(1 - z_min) * pos2[0], z_min * pos1[1] + (1 - z_min) * pos2[1]};;
    std::array<double, 2> char_vec = {(pos[0]-pos_zmin[0])*hy,(pos[1]-pos_zmin[1])*hx};

    double norm_char_vec = std::sqrt(char_vec[0] * char_vec[0] + char_vec[1] * char_vec[1]);
    std::array<double, 2> direction;
    if (norm_char_vec > 0.0) {
        direction = { char_vec[0] / norm_char_vec, char_vec[1] / norm_char_vec };
    } else {
        direction = { 0.0, 0.0 };  // Handle division by zero or very small norm_char_vec
    }

    return std::make_pair(min_V, direction);
}


// Define a struct to hold the double value and its associated key. THis i needed to implement a heap for the considered nodes
struct HeapNode {
    double value;
    Position key;

    // Custom comparator for min heap
    bool operator>(const HeapNode& other) const {
        return value > other.value;
    }
};

// Function to simulate OUM_GRRP
void OUM_GRRP(
    double  (*velocity_func)(const std::array<double, 2>&, const std::array<double, 2>&),
    double Gamma,
    const double* const * elevation,
    const std::vector<Position>& init_nodes,
    const std::vector<double>& init_altitudes,
    const std::vector<std::array<double, 2>>& init_chars,
    double hy,
    double hx,
    double hz,
    int H,
    int W,
    const double* const * const * const * wind_field,
    double **grid,
    double ***char_grid
) {
    // Modify elevation to fit the context

    double triang_diameter = std::sqrt(hx * hx + hy * hy); // Diameter of the triangulation
    double max_dist_edge = 2 * Gamma * triang_diameter;    // Parameter controlling the size of the near front of a node

    // Initialize masks
    bool** mask_accepted = alloc_bool(H, W);
    bool** mask_accepted_front = alloc_bool(H, W);
    bool** mask_considered = alloc_bool(H, W);

    // Initialize considered nodes
    std::priority_queue<HeapNode, std::vector<HeapNode>, std::greater<HeapNode>> considered;

    // Initialize the negative altitude to infinity
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            grid[i][j] = std::numeric_limits<double>::infinity();
        }
    }

    // Initialize accepted nodes
    for (size_t i = 0; i < init_nodes.size(); ++i) {
        int y = init_nodes[i][0];
        int x = init_nodes[i][1];
        grid[y][x] = -init_altitudes[i];
        char_grid[y][x][0] = init_chars[i][0];
        char_grid[y][x][1] = init_chars[i][1];
        mask_accepted[y][x] = true;
    }

    std::vector<double> considered_values_tmp;
    std::vector<Position> considered_nodes_tmp;
    for (size_t i = 0; i < init_nodes.size(); ++i) {
        int y = init_nodes[i][0];
        int x = init_nodes[i][1];
        for (const auto& pos : find_neighbors(init_nodes[i], H, W)) {
            if (!mask_accepted[pos[0]][pos[1]]) {
                mask_accepted_front[y][x] = true;
                if (!mask_considered[pos[0]][pos[1]]) {
                    considered_values_tmp.push_back(grid[y][x]);
                    considered_nodes_tmp.push_back(pos);
                    mask_considered[pos[0]][pos[1]] = true;
                }
            }
        }
    }

    // Initialize the considered nodes
    for (size_t i = 0; i < considered_values_tmp.size(); ++i) {
        Position pos = considered_nodes_tmp[i];
        std::vector<Edge> near_front_edges = compute_near_front(pos, mask_accepted_front, max_dist_edge, hy, hx, H, W);
        //find the wind at the position of the node
        double proxy_altitude = std::numeric_limits<double>::infinity();
        for (const Edge& edge : near_front_edges) {
            const Position& pos1 = edge.first;
            const Position& pos2 = edge.second;
            proxy_altitude = std::min(proxy_altitude, std::min(grid[pos1[0]][pos1[1]], grid[pos2[0]][pos2[1]]));
        }
        int altitude_idx = static_cast<int>(std::round(-proxy_altitude / hz));
        std::array<double, 2> wind;
        wind[0] = wind_field[pos[0]][pos[1]][altitude_idx][0];
        wind[1] = wind_field[pos[0]][pos[1]][altitude_idx][1];

        double min_neg_altitude = std::numeric_limits<double>::infinity();
        std::array<double, 2> min_char;

        for (const Edge& edge : near_front_edges) {
            const Position& pos1 = edge.first;
            const Position& pos2 = edge.second;
            double U1 = grid[pos1[0]][pos1[1]];
            double U2 = grid[pos2[0]][pos2[1]];
            std::pair<double, std::array<double, 2>> result = upwind_simplex_update(pos1, pos2, pos, U1, U2, velocity_func, wind, hy, hx);
            if (result.first < min_neg_altitude) {
                min_neg_altitude = result.first;
                min_char = result.second;
            }
        }

        if (min_neg_altitude <= -elevation[pos[0]][pos[1]]){
            grid[pos[0]][pos[1]] = min_neg_altitude;
            char_grid[pos[0]][pos[1]][0] = min_char[0];
            char_grid[pos[0]][pos[1]][1] = min_char[1];
            considered.push({min_neg_altitude, pos});
        } else {
            throw std::runtime_error("ERROR: initialization too close to obstacle");
        }    
    }
    // Main loop
    while (!considered.empty()) {
        
        Position new_acc_pos = considered.top().key;
        // std::cout << "new_acc_pos: " << new_acc_pos[0] << ", " << new_acc_pos[1] <<", "<<grid[new_acc_pos[0]][new_acc_pos[1]]<<std::endl;
        considered.pop();
        mask_considered[new_acc_pos[0]][new_acc_pos[1]] = false;
        mask_accepted[new_acc_pos[0]][new_acc_pos[1]] = true;
        bool flag_accepted_front = false;

        // Update accepted front
        for (const Position& pos : find_neighbors(new_acc_pos, H, W)) {
            if (mask_accepted_front[pos[0]][pos[1]]) {
                bool remove_pos_from_front = true;
                for (const auto& neigh : find_neighbors(pos, H, W)) {
                    if (mask_considered[neigh[0]][neigh[1]]) {
                        remove_pos_from_front = false;
                        break;
                    }
                }
                if (remove_pos_from_front) {
                    mask_accepted_front[pos[0]][pos[1]] = false;
                }
            }
        }

        mask_accepted_front[new_acc_pos[0]][new_acc_pos[1]] = true;

        // Loop over neighbors of new_acc_pos
        for (const Position& pos : find_neighbors(new_acc_pos, H, W)) {
            if (mask_considered[pos[0]][pos[1]]) {
                flag_accepted_front = true;
            }

            if (!mask_considered[pos[0]][pos[1]] && !mask_accepted[pos[0]][pos[1]]) {
                std::vector<Edge> near_front_edges = compute_near_front(pos, mask_accepted_front, max_dist_edge, hy, hx, H, W);
                if (!near_front_edges.empty()){
                    //find the wind at the position of the node
                    double proxy_altitude = std::numeric_limits<double>::infinity();
                    for (const Edge& edge : near_front_edges) {
                        const Position& pos1 = edge.first;
                        const Position& pos2 = edge.second;
                        proxy_altitude = std::min(proxy_altitude, std::min(grid[pos1[0]][pos1[1]], grid[pos2[0]][pos2[1]]));
                    }
                    std::array<double, 2> wind;
                    wind[0] = wind_field[pos[0]][pos[1]][static_cast<int>(-proxy_altitude / hz+0.5)][0];
                    wind[1] = wind_field[pos[0]][pos[1]][static_cast<int>(-proxy_altitude / hz+0.5)][1];
                    
                    double min_neg_altitude = std::numeric_limits<double>::infinity();
                    std::array<double, 2> min_char;
                    for (size_t r = 0; r < near_front_edges.size(); ++r){
                        const Edge& edge = near_front_edges[r];
                        const Position& pos1 = edge.first;
                        const Position& pos2 = edge.second;
                        double U1 = grid[pos1[0]][pos1[1]];
                        double U2 = grid[pos2[0]][pos2[1]];
                        std::pair<double, std::array<double, 2>> result_upwind = upwind_simplex_update(pos1, pos2, pos, U1, U2, velocity_func, wind, hy, hx);
                        if (result_upwind.first < min_neg_altitude) {
                            min_neg_altitude = result_upwind.first;
                            min_char = result_upwind.second;
                        }
                    }

                    if (min_neg_altitude < -elevation[pos[0]][pos[1]]) {
                        considered.push({min_neg_altitude, pos});
                        mask_considered[pos[0]][pos[1]] = true;
                        grid[pos[0]][pos[1]] = min_neg_altitude;
                        char_grid[pos[0]][pos[1]][0]=min_char[0];
                        char_grid[pos[0]][pos[1]][1]=min_char[1];
                        flag_accepted_front = true;
                    }
                }
            }
        }

        if (!flag_accepted_front) {
            mask_accepted_front[new_acc_pos[0]][new_acc_pos[1]] = false;
        }
    }
   
   //free the allocated memory
    free_bool(mask_accepted, H);
    free_bool(mask_accepted_front, H);
    free_bool(mask_considered, H);


}

int main(int argc, char* argv[]) {       

    bool verbose = false;
    std::string wind_filename;
    std::string elevation_filename;
    std::string initialization_filename;
    std::string output_alt_filename;
    std::string output_char_filename;
    int H, W; // Height and width of the grid
    int Z; // Number of wind layers in the vertical direction
    double h, hz; // Grid spacing (the grid is assumed to be square) and vertical spacing
    // Parse command-line arguments
    for (int i = 1; i < argc; i += 2) {
        std::string arg = argv[i];
        if (arg == "-h") {
            h = std::stod(argv[i + 1]);
        } else if (arg == "-hz") {
            hz = std::stod(argv[i + 1]);
        } else if (arg == "-H") {
            H = std::stoi(argv[i + 1]);
        } else if (arg == "-Z") {
            Z = std::stoi(argv[i + 1]);
        } else if (arg == "-W") {
            W = std::stoi(argv[i + 1]);
        } else if (arg == "-fwind") {
            wind_filename = argv[i + 1];
        } else if (arg == "-finit") {
            initialization_filename = argv[i + 1];
        } else if (arg == "-felev") {
            elevation_filename = argv[i + 1];
        } else if (arg == "-out_alt") {
            output_alt_filename = argv[i + 1];
        } else if (arg == "-out_char") {
            output_char_filename = argv[i + 1];
        } else if (arg == "-v"){
            verbose = true;
        } else {
            showUsage();
            return 1;
        }
    }
    if(verbose){
        std::cout << "Glikonal-G algorithm"  << std::endl;
        std::cout << "h: " << h << std::endl;
        std::cout << "hz: " << hz << std::endl;
        std::cout << "H: " << H << std::endl;
        std::cout << "W: " << W << std::endl;
        std::cout << "Z: " << Z << std::endl;
        std::cout << "wind_filename: " << wind_filename << std::endl;
        std::cout << "elevation_filename: " << elevation_filename << std::endl;
        std::cout << "output_alt_filename: " << output_alt_filename << std::endl;
        std::cout << "output_char_filename: " << output_char_filename << std::endl;
        std::cout <<std::endl<< "Reading the input data from files..."  << std::endl;
    }

    //allocating the elevation profile matrix
    double** elevation = alloc_matrix(H, W);

    //reading the elevation profile
    std::ifstream elevation_file(elevation_filename);
    if (!elevation_file) {
        std::cerr << "Error opening elevation file." << std::endl;
        // Free allocated memory in case of error
        for (int i = 0; i < H; ++i) {
            free(elevation[i]);
        }
        free(elevation);
        return 1;
    }

    // Read the elements into the matrix
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
           elevation_file>> elevation[i][j];
        }
    }
    elevation_file.close();

    //allocating the wind vector field
    double**** wind_field = alloc_tensor4(H, W, Z, 2);

    

    //reading the wind vector field from file 

    std::ifstream wind_file(wind_filename);
    if (!wind_file) {
        std::cerr << "Error opening wind file." << std::endl;
        // Free allocated memory in case of error
        free_matrix(elevation, H);
        free_tensor4(wind_field, H, W, Z);
        return 1;
    }
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            for (int k = 0; k < Z; ++k) {
                wind_file >> wind_field[i][j][k][0];
                wind_file >> wind_field[i][j][k][1];
            }
        }
    }
    wind_file.close();

    //reading the initial conditions from file
    std::ifstream init_file(initialization_filename);
    if (!init_file) {
        std::cerr << "Error opening initialization file." << std::endl;
        // Free allocated memory in case of error
        free_matrix(elevation, H);
        free_tensor4(wind_field, H, W, Z);
        return 1;
    }
    std::string line;
    std::vector<double> numbers; // vector to store the parsed numbers contained in a single line of the file


    std::vector<Position> init_nodes;
    std::vector<double> init_altitudes;
    std::vector<std::array<double, 2>> init_chars;

    // Read the file line by line
    while (std::getline(init_file, line)) {
        std::istringstream iss(line);
        double number;

        // Parse each number in the line
        while (iss >> number) {
            numbers.push_back(number);
        }
        init_nodes.push_back({static_cast<int>(numbers[0]+0.5), static_cast<int>(numbers[1]+0.5)});
        init_altitudes.push_back(numbers[2]);
        init_chars.push_back({numbers[3], numbers[4]});
        numbers.clear();
    }
 
    init_file.close();
    if(verbose){
        std::cout << "Initializing the data structures..." << std::endl;
    }
    //running the Glikonal-G algorithm
    double** grid = alloc_matrix(H, W);
    double*** char_grid = alloc_tensor3(H, W, 2);
    double max_wind=0.0;
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            for (int k = 0; k < Z; ++k) {
                max_wind=std::max(max_wind,std::sqrt(wind_field[i][j][k][0]*wind_field[i][j][k][0]+wind_field[i][j][k][1]*wind_field[i][j][k][1]));
            }
        }
    }
    double Gamma = (1.+max_wind)/(1.-max_wind);

     if(verbose){
        std::cout << "Running the algorithm..." << std::endl;
    }
    auto start = std::chrono::high_resolution_clock::now();
    OUM_GRRP(velocity_function, Gamma, elevation, init_nodes, init_altitudes, init_chars, h,h, hz, H, W, wind_field, grid, char_grid);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;
    if(verbose){
        std::cout << "Time to run OUM_GRRP: " << elapsed_seconds.count() << "s" << std::endl;
        std::cout << "Writing the results on file" << std::endl;
    }

//output the grid and the characteristics

//open output file
std::ofstream output_file(output_alt_filename);
if (!output_file) {
    std::cerr << "Error opening output file." << std::endl;
    // Free allocated memory in case of error
    free_matrix(elevation, H);
    free_tensor4(wind_field, H, W, Z);
    free_tensor3(char_grid, H, W);
    free_matrix(grid, H);
    return 1;
}

// Write the grid values to the output file
for (int i = 0; i < H; ++i) {
    for (int j = 0; j < W; ++j) {
        output_file << -grid[i][j] << " ";
    }
    output_file << std::endl;
}
output_file.close();

//open output file for the characteristics
std::ofstream output_char_file(output_char_filename);
if (!output_char_file) {
    std::cerr << "Error opening output file for the characteristics." << std::endl;
    // Free allocated memory in case of error
    free_matrix(elevation, H);
    free_tensor4(wind_field, H, W, Z);
    free_tensor3(char_grid, H, W);
    free_matrix(grid, H);
    return 1;
}
// write characteristics to the output file
for (int i = 0; i < H; ++i) {
    for (int j = 0; j < W; ++j) {
        output_char_file << char_grid[i][j][0] << " " << char_grid[i][j][1] << " "<<std::endl;
    }
}
output_char_file.close();

//freeing the allocated memory
    free_matrix(elevation, H);
    free_tensor4(wind_field, H, W, Z);
    free_tensor3(char_grid, H, W);
    free_matrix(grid, H);
    if(verbose){
        std::cout << "Done! Bye Bye" << std::endl;
    }
    return 0;


}
