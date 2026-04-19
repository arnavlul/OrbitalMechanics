#include <iostream>
#include "eigen/Eigen/Dense"
#include <vector>
#include <fstream>
#include <random>
#include <cmath>
#include <sstream>
#define endl '\n'
using namespace std;
using namespace Eigen;

string DATA_FILE = "./cleaned_data.csv";
string TESTING_FILE = "./test_files/earth_test.csv";
string ERRORS_FILE = "./saved_errors.csv";
string SIMULATION_FILE = "./simulated_data.csv";
double WIGGLE_ROOM = 1.5; // Multiplying max_vec by this number so that orbits don't behave weird if for a sample orbit the number is greater than max_vec

int INPUT_SIZE = 6;
int OUTPUT_SIZE = 6;
int HIDDEN_SIZE = 64;
int BATCH_SIZE = 32;
int NUM_EPOCHS = 20;    
double ALPHA = 0.01;

struct OrbitState{
    double jd; // Julian Date
    double x, y, z, vx, vy, vz;

    void printState(){
        cout << "Coords: [" << x << ", " << y << ", " << z << "]\n";
        cout << "Velocity: [" << vx << ", " << vy << ", " << vz << "]\n";
        return;
   }
};

int sgn(double n){
    if(n > 0) return 1;
    if(n == 0) return 0;
    return -1;
}

vector<OrbitState> read_data(const string &filename){
    cout << "Reading data..." << endl;
    vector<OrbitState> data;
    ifstream file(filename);
    
    if(!file.is_open()){
        cerr << "File " << filename << " didn't open";
        return data;
    }
    
    string line;

    // int count = 0;

    while(getline(file, line)){
        stringstream ss(line);
        string col;

        OrbitState state;
        try{
            getline(ss, col, ','); state.jd = stod(col);
            getline(ss, col, ','); state.x = stod(col);
            getline(ss, col, ','); state.y = stod(col);
            getline(ss, col, ','); state.z = stod(col);
            getline(ss, col, ','); state.vx = stod(col);
            getline(ss, col, ','); state.vy = stod(col);
            getline(ss, col, ','); state.vz = stod(col);
            data.push_back(state);
            // count++;
        }
        catch(exception &e){
            cout << "Exception: " << e.what();
            continue;
        }
    }

    // cout << "Lines pushed: " << count << endl;
    cout << "Data read." << endl;
    return data;

}

void normalise_data(vector<OrbitState> &data, VectorXd &max_vec){

    cout << "Normalising data..." << endl;

    max_vec = VectorXd::Zero(6);

    for(const auto &i : data){
        max_vec(0) = max(max_vec(0), abs(i.x));
        max_vec(1) = max(max_vec(1), abs(i.y));
        max_vec(2) = max(max_vec(2), abs(i.z));
        max_vec(3) = max(max_vec(3), abs(i.vx));
        max_vec(4) = max(max_vec(4), abs(i.vy));
        max_vec(5) = max(max_vec(5), abs(i.vz));
        
    }

    max_vec *= WIGGLE_ROOM;

    for(auto &i : data){
        i.x /= max_vec(0);
        i.y /= max_vec(1);
        i.z /= max_vec(2);
        i.vx /= max_vec(3);
        i.vy /= max_vec(4);
        i.vz /= max_vec(5);
    }
    
    cout << "Data Normalised." << endl;

    return;
}

void get_batch(const vector<OrbitState> &data, const vector<int> &indices, int batch_idx, int batch_size, MatrixXd &X, MatrixXd &Y){
    for(int i=0; i<batch_size; i++){
    // Wrap around if current_place is outside indice size
        int current_place = (batch_idx * batch_size + i) % indices.size();

        int global_idx = indices[current_place];

    // Wrap around if global_idx is outside data size
        if(global_idx >= data.size()-1){
            global_idx = 0;
        }

        X(0, i) = data[global_idx].x;
        X(1, i) = data[global_idx].y;
        X(2, i) = data[global_idx].z;
        X(3, i) = data[global_idx].vx;
        X(4, i) = data[global_idx].vy;
        X(5, i) = data[global_idx].vz;

        Y(0, i) = data[global_idx + 1].x;
        Y(1, i) = data[global_idx + 1].y;
        Y(2, i) = data[global_idx + 1].z;
        Y(3, i) = data[global_idx + 1].vx;
        Y(4, i) = data[global_idx + 1].vy;
        Y(5, i) = data[global_idx + 1].vz;
    }

    return;
}

void initialise_weights(int input_size, int hidden_size, int output_size, MatrixXd &w1, VectorXd &b1, MatrixXd &w2, VectorXd &b2){
    srand(42);

    double limit1 = sqrt((6.0/(input_size + hidden_size)));
    double limit2 = sqrt((6.0 / (hidden_size + output_size)));

    w1 = MatrixXd::Random(hidden_size, input_size) * limit1;
    b1 = VectorXd::Zero(hidden_size);
    w2 = MatrixXd::Random(output_size, hidden_size) * limit2;
    b2 = VectorXd::Zero(output_size);

    cout << "Initialised weights." << endl;
    return;
}

MatrixXd activation_tanh(const MatrixXd &Z){
    return Z.array().tanh();
}

MatrixXd forward_pass(const MatrixXd &X, const MatrixXd &w1, const VectorXd &b1, const MatrixXd &w2, const VectorXd &b2, MatrixXd &hidden_cache){
    hidden_cache = (w1*X).colwise() + b1;
    hidden_cache = activation_tanh(hidden_cache);
    return (w2*hidden_cache).colwise() + b2;
}

double compute_loss(const MatrixXd &Y_pred, const MatrixXd &Y_target){
    double mse;

    MatrixXd difference = Y_pred - Y_target;
    mse = difference.squaredNorm();
    mse /= BATCH_SIZE;
    return mse;
}

void backward_pass(const MatrixXd &X, const MatrixXd &Y_pred, const MatrixXd &Y_target, const MatrixXd &hidden_act, const MatrixXd &w2, MatrixXd &dw1, VectorXd &db1, MatrixXd &dw2, VectorXd &db2){
    MatrixXd dY_p, dJ, dH;

    dY_p = (2.0/BATCH_SIZE)*(Y_pred - Y_target);
    dw2 = dY_p * hidden_act.transpose();

    dH = w2.transpose() * dY_p;
    dJ = dH.cwiseProduct((1.0-hidden_act.array().square()).matrix());
    dw1 = dJ * X.transpose();

    db2 = dY_p.rowwise().sum();
    db1 = dJ.rowwise().sum();
    return;
}

void update_weights(MatrixXd &w1, VectorXd &b1, MatrixXd &w2, VectorXd &b2, const MatrixXd &dw1, const VectorXd &db1, const MatrixXd &dw2, const VectorXd &db2){
    w1 -= ALPHA*dw1;
    w2 -= ALPHA*dw2;
    b1 -= ALPHA*db1;
    b2 -= ALPHA*db2;
    return;
}

OrbitState denormalise_data(const MatrixXd &norm_data, const VectorXd &max_vec){
    OrbitState os;
    os.jd = 0.0;
    os.x = norm_data(0,0) * max_vec(0);
    os.y = norm_data(1,0) * max_vec(1);
    os.z = norm_data(2,0) * max_vec(2);
    os.vx = norm_data(3,0) * max_vec(3);
    os.vy = norm_data(4,0) * max_vec(4);
    os.vz = norm_data(5,0) * max_vec(5);
    return os;
}

vector<OrbitState> simulate_data(const string &filename, const OrbitState &start_state, int steps, double dt,const MatrixXd &w1, const VectorXd &b1, const MatrixXd &w2, const VectorXd &b2, const VectorXd &max_vec){
    MatrixXd X(6,1);
    X(0,0) = start_state.x / max_vec(0);
    X(1,0) = start_state.y / max_vec(1);
    X(2,0) = start_state.z / max_vec(2);
    X(3,0) = start_state.vx / max_vec(3);
    X(4,0) = start_state.vy / max_vec(4);
    X(5,0) = start_state.vz / max_vec(5);
    

    vector<OrbitState> states;

    ofstream file(filename);
    if(!file.is_open()){
        cerr << "Simulation file not opening.";
        return states;
    }

    file << "JD,x,y,z,vx,vy,vz" << endl;
    double current_time = start_state.jd;

    for(int i=0; i<steps; i++){
    // Forward Passes
        X = (w1*X).colwise() + b1;
        X = activation_tanh(X);
        X = (w2*X).colwise() + b2;
        X = X.array().max(-1.0).min(1.0).matrix();
    // Saving Data
        VectorXd Z = X;
        OrbitState os = denormalise_data(Z, max_vec);
        current_time += dt;
        os.jd = current_time;
        states.push_back(os);

        file << states[i].jd << "," << states[i].x << "," << states[i].y << "," << states[i].z << "," << states[i].vx << "," << states[i].vy << "," << states[i].vz << endl;
    }

    file.close();
    cout << "Simulation saved." << endl;

    return states;
}

double test_error(const string &filename, const MatrixXd &w1, const VectorXd &b1, const MatrixXd &w2, const VectorXd &b2, const VectorXd &max_vec){
    vector<OrbitState> test_set = read_data(filename);

// Normalising Data
    for(auto &i : test_set){
        i.x /= max_vec(0);
        i.y /= max_vec(1);
        i.z /= max_vec(2);
        i.vx /= max_vec(3);
        i.vy /= max_vec(4);
        i.vz /= max_vec(5);
    }

    MatrixXd X(6,1), Y(6,1);
    double MSE = 0;

    for(int i=0; i<test_set.size()-1; i++){
        X(0,0) = test_set[i].x;
        X(1,0) = test_set[i].y;
        X(2,0) = test_set[i].z;
        X(3,0) = test_set[i].vx;
        X(4,0) = test_set[i].vy;
        X(5,0) = test_set[i].vz;

        Y(0,0) = test_set[i+1].x;
        Y(1,0) = test_set[i+1].y;
        Y(2,0) = test_set[i+1].z;
        Y(3,0) = test_set[i+1].vx;
        Y(4,0) = test_set[i+1].vy;
        Y(5,0) = test_set[i+1].vz;

        X = (w1 * X).colwise() + b1;
        X = activation_tanh(X);
        X = (w2 * X).colwise() + b2;

        X = (X-Y).array().square();
        MSE += X.sum();
    }
    MSE /= test_set.size()-1;
    return MSE;

}

void save_errors(const string &filename, const vector<double> errors){
    ofstream file(filename);
    
    if(!file.is_open()){
        cerr << "Error file couldn't open" << endl;
        return;
    }

    file << "Epoch,Error" << endl;

    for(int i=0; i<errors.size(); i++){
        file << (i+1) << "," << errors[i] << endl;
    }

    file.close();
    cout << "Errors saved." << endl;
    return;
}

int main(){

// Reading Data
    vector<OrbitState> data = read_data(DATA_FILE); 

// Normalising Data
    VectorXd max_vec;
    normalise_data(data, max_vec);

// Helper variables' init
    vector<int> indices(data.size());
    for(int i=0; i<indices.size(); i++) indices[i] = i;

    int num_batches = indices.size() / BATCH_SIZE;

// Main loop
    random_device rd;
    mt19937 rng(rd());
    
// Initialising Weights
    MatrixXd w1, w2; VectorXd b1, b2;
    initialise_weights(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, w1, b1, w2, b2);

    vector<double> error_progression;

    for(int i=0; i<NUM_EPOCHS; i++){
        
        shuffle(indices.begin(), indices.end(), rng);
        
        for(int j=0; j<num_batches; j++){
        // Getting Batches
            MatrixXd X(6, BATCH_SIZE), Y_target(6, BATCH_SIZE);
            get_batch(data, indices, j, BATCH_SIZE, X, Y_target);


        // Forward Pass
            MatrixXd hidden_cache, Y_pred;
            Y_pred = forward_pass(X, w1, b1, w2, b2, hidden_cache);

        // Computing loss
            double mse = compute_loss(Y_pred, Y_target);

        // Backpropogation
            MatrixXd dw1, dw2; VectorXd db1, db2;
            backward_pass(X, Y_pred, Y_target, hidden_cache, w2, dw1, db1, dw2, db2);
            update_weights(w1, b1, w2, b2, dw1, db1, dw2, db2);

        }

        double MSE = test_error(TESTING_FILE, w1, b1, w2, b2, max_vec);
        error_progression.push_back(MSE);
        cout << "Epoch " << i+1 << " complete. MSE: " << MSE << endl;
    }

    cout << "All epochs completed" << endl;

// Saving Errors to see how they change over epochs
    save_errors(ERRORS_FILE, error_progression);

// Simulating a state
    double learned_dt = data[1].jd - data[0].jd;
    OrbitState os;
    os.jd = 2449718.500000000;
    os.x = -2.595232416743753E+07;
    os.y = 1.447959489186955E+08;
    os.z = 1.015388480179012E+03;
    os.vx = -2.982057650057118E+01;
    os.vy = -5.367390412066213E+00;
    os.vz = -8.642933998290747E-04;

 

    simulate_data(SIMULATION_FILE, os, 1000, learned_dt, w1, b1, w2, b2, max_vec);
    


    return 0;
}