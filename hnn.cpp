#include <iostream>
#include <vector>
#include "eigen/Eigen/Dense"
#include <vector>
#include <fstream>
#include <sstream>
#include <random>
#include <cmath>
#include <iomanip>
#define endl '\n'
using namespace std;
using namespace Eigen;

const string DATA_FILE = "./made_files/cleaned_data.csv";
const string TEST_FILE;
const string ERRORS_FILE = "hnn_errors.csv";
const string SIMULATION_FILE = "hnn_sim.csv";

const int NUM_EPOCHS = 200;
const int INPUT_SIZE = 6;
const int OUTPUT_SIZE = 1;
const int BATCH_SIZE = 512;
const int NEURONS_H1 = 128;
const int NEURONS_H2 = 128;
const double epsilon = 0.0000001;
double alpha = 0.0001;
double lambda = 0.0;
double dt = 1; // Time-step for simulation

const double AU_to_M = 149597870700.0;
const double VELOCITY_DIVISOR = 1731.45683681;
const double seconds_in_day = 86400.0;

struct OrbitState{
    double jd; // Julian Date
    double x, y, z, vx, vy, vz, ax, ay, az;

    void printState(){
        cout << "Coords: [" << x << ", " << y << ", " << z << "]\n";
        cout << "Velocity: [" << vx << ", " << vy << ", " << vz << "]\n";
        return;
   }
};

vector<OrbitState> read_data(const string filename){
    cout << "Reading Data" << endl;
    ifstream file(filename);

    vector<OrbitState> v;

    if(!file.is_open()){
        cerr << "Couldn't open file";
        return v;
    }

    string line;

    while(getline(file, line)){
        stringstream ss(line);
        string col;

        OrbitState os;
        try{
            getline(ss, col, ','); os.jd = stod(col);
            getline(ss, col, ','); os.x = stod(col);
            getline(ss, col, ','); os.y = stod(col);
            getline(ss, col, ','); os.z = stod(col);
            getline(ss, col, ','); os.vx = stod(col);
            getline(ss, col, ','); os.vy = stod(col);
            getline(ss, col, ','); os.vz = stod(col);
            v.push_back(os);
        }
        catch(exception &e){
            cout << "Exception: " << e.what() << endl;
            continue;
        }
    }
    cout << v.size() << " entries added." << endl;
    return v;
}

void normalise_data(vector<OrbitState> &data, VectorXd &m, VectorXd &s){
    VectorXd means = VectorXd::Zero(9);
    
    // 1. Calculate Means (standard)
    for(auto &i : data){
        means(0) += i.x; means(1) += i.y; means(2) += i.z;
        means(3) += i.vx; means(4) += i.vy; means(5) += i.vz;
        means(6) += i.ax; means(7) += i.ay; means(8) += i.az;
    }
    means /= data.size();

    // 2. Calculate "Isotropic" Standard Deviation
    // We pool the variance of X, Y, Z together.
    double var_pos = 0.0, var_vel = 0.0, var_acc = 0.0;
    
    for(auto &i : data){
        // Sum squares for Position group
        var_pos += pow(i.x - means(0), 2) + pow(i.y - means(1), 2) + pow(i.z - means(2), 2);
        // Sum squares for Velocity group
        var_vel += pow(i.vx - means(3), 2) + pow(i.vy - means(4), 2) + pow(i.vz - means(5), 2);
        // Sum squares for Acceleration group
        var_acc += pow(i.ax - means(6), 2) + pow(i.ay - means(7), 2) + pow(i.az - means(8), 2);
    }

    // Divide by (N * 3) because we summed 3 dimensions
    double std_pos = sqrt(var_pos / (data.size() * 3.0));
    double std_vel = sqrt(var_vel / (data.size() * 3.0));
    double std_acc = sqrt(var_acc / (data.size() * 3.0));

    // 3. Store in the 's' vector (Broadcast the values)
    s = VectorXd::Zero(9);
    s.segment(0, 3).setConstant(std_pos); // s(0)=s(1)=s(2) = std_pos
    s.segment(3, 3).setConstant(std_vel); // s(3)=s(4)=s(5) = std_vel
    s.segment(6, 3).setConstant(std_acc); // s(6)=s(7)=s(8) = std_acc

    // 4. Apply Normalization
    for(auto &i : data){
        i.x = (i.x - means(0)) / s(0); i.y = (i.y - means(1)) / s(1); i.z = (i.z - means(2)) / s(2);
        i.vx = (i.vx - means(3)) / s(3); i.vy = (i.vy - means(4)) / s(4); i.vz = (i.vz - means(5)) / s(5);
        i.ax = (i.ax - means(6)) / s(6); i.ay = (i.ay - means(7)) / s(7); i.az = (i.az - means(8)) / s(8);
    }

    m = means;
}

void scale_data(vector<OrbitState> &v){
    cout << "Scaling data..." << endl;

    for(auto &i : v){
        i.x /= AU_to_M;
        i.y /= AU_to_M;
        i.z /= AU_to_M;
        i.vx /= VELOCITY_DIVISOR;
        i.vy /= VELOCITY_DIVISOR;
        i.vz /= VELOCITY_DIVISOR;
    }

    cout << "Data Scaled." << endl;
}

void initialise_weights(MatrixXd &W_1, MatrixXd &W_2, MatrixXd &W_3, VectorXd &B_1, VectorXd &B_2, VectorXd &B_3){
    srand(42);

    double limit1 = sqrt(6.0 / (INPUT_SIZE + NEURONS_H1)); 
    double limit2 = sqrt(6.0 / (NEURONS_H1 + NEURONS_H2));
    double limit3 = sqrt(6.0 / (NEURONS_H2 + OUTPUT_SIZE));

    W_1 = MatrixXd::Random(NEURONS_H1, INPUT_SIZE) * limit1;
    W_2 = MatrixXd::Random(NEURONS_H2, NEURONS_H1) * limit2;
    W_3 = MatrixXd::Random(OUTPUT_SIZE, NEURONS_H2) * limit3;
    B_1 = VectorXd::Zero(NEURONS_H1);
    B_2 = VectorXd::Zero(NEURONS_H2);
    B_3 = VectorXd::Zero(OUTPUT_SIZE);

    return;
}

vector<int> precompute_acceleration(vector<OrbitState> &data){
    
    cout << "Calculating Accelerations..." << endl;

    vector<int> validIndices;

    for(int i=1; i<data.size()-1; i++){
        double dt = data[i+1].jd - data[i-1].jd;

        if(dt >= 3.0){
            data[i].ax = data[i].ay = data[i].az = 0;
            continue;
        }

        data[i].ax = (data[i+1].vx - data[i-1].vx) / dt;
        data[i].ay = (data[i+1].vy - data[i-1].vy) / dt;
        data[i].az = (data[i+1].vz - data[i-1].vz) / dt;

        validIndices.push_back(i);
    }

    cout << "Accelerations calcualted. Valid indices derived." << endl;
    return validIndices;
}

MatrixXd activation_tanh(const MatrixXd Z){
    return Z.array().tanh();
}

MatrixXd forward_pass(const MatrixXd I, const MatrixXd W_1, const VectorXd B_1, const MatrixXd W_2, const VectorXd B_2, const MatrixXd W_3, const VectorXd B_3, MatrixXd &A_1, MatrixXd &A_2){

    MatrixXd Z_1 = (W_1 * I).colwise() + B_1;
    A_1 = activation_tanh(Z_1);
    MatrixXd Z_2 = (W_2 * A_1).colwise() + B_2;
    A_2 = activation_tanh(Z_2);
    MatrixXd H = (W_3 * A_2).colwise() + B_3;
    
    return H;
}

void backprop_logic(const MatrixXd &I, const MatrixXd &E, const MatrixXd &A_1, const MatrixXd &A_2,const MatrixXd &W_2, const MatrixXd &W_3,
MatrixXd &dW_1, MatrixXd &dW_2, MatrixXd &dW_3, VectorXd &dB_1, VectorXd &dB_2, VectorXd &dB_3){
        
    MatrixXd dH = E/(2*epsilon);

    dW_3 = (dH) * (A_2).transpose();  
    
    MatrixXd dA_2 = W_3.transpose() * dH;
    MatrixXd dZ_2 = dA_2.cwiseProduct((1.0-A_2.array().square()).matrix());
    dW_2 = dZ_2 * A_1.transpose();

    MatrixXd dA_1 = W_2.transpose() * dZ_2;
    MatrixXd dZ_1 = dA_1.cwiseProduct((1.0-A_1.array().square()).matrix());
    dW_1 = dZ_1 * I.transpose();

    dB_1 = dZ_1.rowwise().sum(); // Sum over the whole batch
    dB_2 = dZ_2.rowwise().sum();
    dB_3 = dH.rowwise().sum();
    return;
}

void compute_physics_gradients(const MatrixXd &I, const MatrixXd &T, const MatrixXd &W_1, const VectorXd &B_1, const MatrixXd &W_2, const VectorXd &B_2, const MatrixXd &W_3, const VectorXd &B_3, 
MatrixXd &dW1_total, VectorXd &dB1_total, MatrixXd &dW2_total, VectorXd &dB2_total, MatrixXd &dW3_total, VectorXd &dB3_total, double &batch_loss, double &mse_loss, double &energy_loss){

    dW1_total.setZero(); dW2_total.setZero(); dW3_total.setZero(); // To remove any (possible) leftover values from previous epochs
    dB1_total.setZero(); dB2_total.setZero(); dB3_total.setZero();
    batch_loss = mse_loss = energy_loss = 0.0;
    
    vector<MatrixXd> store_pos(6), store_neg(6), store_A1_pos(6), store_A1_neg(6), store_A2_pos(6), store_A2_neg(6), store_slopes(6);
    MatrixXd total_drift = MatrixXd::Zero(1,BATCH_SIZE);

    for(int i=0; i<6; i++){
        MatrixXd A_1plus, A_1minus, A_2plus, A_2minus, H_plus, H_minus, slope, error;

        MatrixXd temp_pos = I; temp_pos.row(i) = temp_pos.row(i).array() + epsilon;
        H_plus = forward_pass(temp_pos, W_1, B_1, W_2, B_2, W_3, B_3, A_1plus, A_2plus);

        MatrixXd temp_neg = I; temp_neg.row(i) = temp_neg.row(i).array() - epsilon;
        H_minus = forward_pass(temp_neg, W_1, B_1, W_2, B_2, W_3, B_3, A_1minus, A_2minus);

        slope = (H_plus - H_minus).array() / (2*epsilon);
        
        store_slopes[i] = slope;
        store_pos[i] = temp_pos; store_neg[i] = temp_neg;
        store_A1_pos[i] = A_1plus; store_A1_neg[i] = A_1minus;
        store_A2_pos[i] = A_2plus; store_A2_neg[i] = A_2minus;

    // dH/dT = dH/dz_i * dz_i/dt (Chain Rule)
        total_drift.array() += slope.array() * T.row(i).array();
    }

    energy_loss = lambda * total_drift.array().square().sum();
    batch_loss += energy_loss;
    
    for(int i=0; i<6; i++){
        MatrixXd error_mse, error_energy, total_error;
        MatrixXd dW1_temp, dW2_temp, dW3_temp;
        VectorXd dB1_temp, dB2_temp, dB3_temp;

        if(i<3.0){
            error_mse = 2*(store_slopes[i] - (-T.row(i+3)));
        }
        else{
            error_mse = 2*(store_slopes[i] - (T.row(i-3)));
        }
        
        double dim_mse = (error_mse / 2.0).array().square().sum();
        mse_loss += dim_mse;
        batch_loss += dim_mse; // Adding all the individual losses (Loss = (Pred - FInal)^2, error is derivative of loss)
        
        error_energy = 2 * lambda * total_drift.array() * T.row(i).array();
        
        total_error = error_mse + error_energy;
            
        backprop_logic(store_pos[i], total_error, store_A1_pos[i], store_A2_pos[i], W_2, W_3, dW1_temp, dW2_temp, dW3_temp, dB1_temp, dB2_temp, dB3_temp);
        dW1_total += dW1_temp; dW2_total += dW2_temp; dW3_total += dW3_temp;
        dB1_total += dB1_temp; dB2_total += dB2_temp; dB3_total += dB3_temp;
        
        // dL/dH_-ve = (error) E * (-1/(2*epsilon))
        backprop_logic(store_neg[i], -total_error, store_A1_neg[i], store_A2_neg[i], W_2, W_3, dW1_temp, dW2_temp, dW3_temp, dB1_temp, dB2_temp, dB3_temp);
        dW1_total += dW1_temp; dW2_total += dW2_temp; dW3_total += dW3_temp;
        dB1_total += dB1_temp; dB2_total += dB2_temp; dB3_total += dB3_temp;
    }

    
    batch_loss /= (BATCH_SIZE * 6);
    mse_loss /= (BATCH_SIZE * 6);
    energy_loss /= (BATCH_SIZE * 6);
}

void get_batches(const vector<OrbitState> &data, const vector<int> &indices, const int batch_idx, MatrixXd &I, MatrixXd &T){
    int indices_size = indices.size();
    int data_size = data.size();
    for(int i=0; i<BATCH_SIZE; i++){
        int current_place = (BATCH_SIZE * batch_idx + i) % indices_size;
        int global_idx = indices[current_place];

        if(global_idx < 1 || global_idx > data_size-2) global_idx = 1;

        I(0, i) = data[global_idx].x;
        I(1, i) = data[global_idx].y;
        I(2, i) = data[global_idx].z;
        I(3, i) = data[global_idx].vx;
        I(4, i) = data[global_idx].vy;
        I(5, i) = data[global_idx].vz;

        T(0, i) = data[global_idx].vx;
        T(1, i) = data[global_idx].vy;
        T(2, i) = data[global_idx].vz;
        T(3, i) = data[global_idx].ax;
        T(4, i) = data[global_idx].ay;
        T(5, i) = data[global_idx].az;
    }

    return;
}

void update_weights(MatrixXd &W_1, MatrixXd &W_2, MatrixXd &W_3, VectorXd &B_1, VectorXd &B_2, VectorXd &B_3, 
const MatrixXd &dW1, const MatrixXd &dW2, const MatrixXd &dW3, const VectorXd &dB1, const VectorXd &dB2, const VectorXd &dB3){

    W_1 -= alpha * dW1;
    W_2 -= alpha * dW2;
    W_3 -= alpha * dW3;
    B_1 -= alpha * dB1;
    B_2 -= alpha * dB2;
    B_3 -= alpha * dB3;
    return;
}

VectorXd sim_helper(const OrbitState &state, const MatrixXd &W1, const MatrixXd &W2, const MatrixXd &W3, const VectorXd &B1, const VectorXd &B2, const VectorXd &B3){

    VectorXd dynamics(6);
    
    MatrixXd I(6,1); // Initialising state Matrix
    I(0,0) = state.x;
    I(1,0) = state.y;
    I(2,0) = state.z;
    I(3,0) = state.vx;
    I(4,0) = state.vy;
    I(5,0) = state.vz;

    
    for(int i=0; i<6; i++){
        MatrixXd A1_dummy, A2_dummy, H_plus, H_minus, I_plus = I, I_minus = I; // Need A1_dummy and A2_dummy for forward pass funcn arguments
        
        I_plus(i,0) += epsilon;
        I_minus(i, 0) -= epsilon;

        H_plus = forward_pass(I_plus, W1, B1, W2, B2, W3, B3, A1_dummy, A2_dummy); 
        H_minus = forward_pass(I_minus, W1, B1, W2, B2, W3, B3, A1_dummy, A2_dummy);

        double slope = (H_plus.value() - H_minus.value())/(2*epsilon);

        if(i<3) dynamics(i+3) = -slope;
        else dynamics(i-3) = slope;
    }

    return dynamics;
}

void simulate_data(const string &filename, OrbitState &initial, int steps, const MatrixXd &W1, const MatrixXd &W2, const MatrixXd &W3, const VectorXd &B1, const VectorXd &B2, const VectorXd &B3, const VectorXd &means, const VectorXd &std_dev){
    
    ofstream file(filename);
    if(!file.is_open()){
        cerr << "Simulation File could not be opened";
        return;
    }

    file.precision(15);
    file << scientific;

    file << "jd,x,y,z,vx,vy,vz,ax,ay,az" << endl;
   
    // Normalising initial
    initial.x = (initial.x - means(0)) / std_dev(0);
    initial.y = (initial.y - means(1)) / std_dev(1);
    initial.z = (initial.z - means(2)) / std_dev(2);
    initial.vx = (initial.vx - means(3)) / std_dev(3);
    initial.vy = (initial.vy - means(4)) / std_dev(4);
    initial.vz = (initial.vz - means(5)) / std_dev(5);

    
    vector<OrbitState> sim;
    sim.push_back(initial);

    double r_v = std_dev(6) / std_dev(3);
    double r_x = (std_dev(3) / std_dev(0)) * seconds_in_day;

    // Initial Acceleration Calculation
    VectorXd dynamics = sim_helper(initial, W1, W2, W3, B1, B2, B3);
    double ax = dynamics(3);
    double ay = dynamics(4);
    double az = dynamics(5);

    for(int i=0; i<steps; i++){
        OrbitState os = sim[sim.size()-1];

        os.vx += 0.5 * ax * dt * r_v;
        os.vy += 0.5 * ay * dt * r_v;
        os.vz += 0.5 * az * dt * r_v;

        os.x += os.vx * dt * r_x;
        os.y += os.vy * dt * r_x;
        os.z += os.vz * dt * r_x;

        dynamics = sim_helper(os, W1, W2, W3, B1, B2, B3);

        os.ax = dynamics(3);
        os.ay = dynamics(4);
        os.az = dynamics(5);

        os.vx += 0.5 * os.ax * dt * r_v;
        os.vy += 0.5 * os.ay * dt * r_v;
        os.vz += 0.5 * os.az * dt * r_v;

        ax = os.ax; ay = os.ay; az = os.az;
        os.jd++;

        sim.push_back(os);
    }

    // De-normalising
    for(auto &i : sim){
        i.x = (i.x * std_dev(0)) + means(0);
        i.y = (i.y * std_dev(1)) + means(1);
        i.z = (i.z * std_dev(2)) + means(2);
        i.vx = (i.vx * std_dev(3)) + means(3);
        i.vy = (i.vy * std_dev(4)) + means(4);
        i.vz = (i.vz * std_dev(5)) + means(5);
        i.ax = (i.ax * std_dev(6)) + means(6);
        i.ay = (i.ay * std_dev(7)) + means(7);
        i.az = (i.az * std_dev(8)) + means(8);
    }

    // Writing Simulated Data
    cout << "Writing Simulated Data to File" << endl;
    for(const auto& i : sim){
        file << i.jd << "," << i.x << "," << i.y << "," << i.z << "," << i.vx << "," << i.vy << "," << i.vz << "," << i.ax << "," << i.ay << "," << i.az << endl;
    }

    cout << "Simulation Finished" << endl;

}

OrbitState string_to_orbitstate(const string &data){
    OrbitState os;
    stringstream ss(data);
    string col;

    getline(ss, col, ','); os.jd = stod(col);  
    getline(ss, col, ','); os.x = stod(col);
    getline(ss, col, ','); os.y = stod(col);
    getline(ss, col, ','); os.z = stod(col);
    getline(ss, col, ','); os.vx = stod(col);
    getline(ss, col, ','); os.vy = stod(col);
    getline(ss, col, ','); os.vz = stod(col);

    return os;
}

void save_error_to_file(const string &filename, const vector<vector<double>> &data){
    ofstream file(filename);
    if(!file.is_open()){
        cerr << "Error file not opening" << endl;
        return;
    }

    file << "Epoch,Total Error,MSE Error,Energy Error" << endl;
    file << scientific << setprecision(10); // Use scientific notation for all values

    for(int i=0; i<data.size(); i++){
        file << i+1 << "," << data[i][0] << "," << data[i][1] << "," << data[i][2] << endl;
    }
}

void get_hyperparameters(const int &epoch, double &alpha, double &lambda){
    if(epoch < 100) alpha = 0.001;
    else if(epoch < 150) alpha = 0.0005;
    else alpha = 0.0001;

    if(epoch < 70) lambda = 0.0;
    else lambda = 1.0;
    return;
}

int main(){

    vector<OrbitState> data = read_data(DATA_FILE); 
    
    // Initialising weights
    MatrixXd I(6, BATCH_SIZE), Z_1 (NEURONS_H1, BATCH_SIZE), Z_2(NEURONS_H2, BATCH_SIZE), A_1(NEURONS_H1, BATCH_SIZE), A_2(NEURONS_H2, BATCH_SIZE), H_out (1,BATCH_SIZE), W_1(NEURONS_H1, 6), W_2 (NEURONS_H2, NEURONS_H1), W_3 (1, NEURONS_H2);
    VectorXd B_1(NEURONS_H1), B_2(NEURONS_H2), B_3(1);
    initialise_weights(W_1, W_2, W_3, B_1, B_2, B_3); // Initialises according to Xavier Initialisation
   
    // RNG to shuffle indices
    random_device rd;
    mt19937 rng(rd());
    
    vector<int> indices = precompute_acceleration(data); // Initialising random indices set
    int num_batches = indices.size() / BATCH_SIZE;
    
    // Normalising Data
    VectorXd means, std_dev;
    normalise_data(data, means, std_dev);

    // Initialising Vector to Store MSE
    vector<vector<double>> loss_over_time;

    for(int i=1; i<=NUM_EPOCHS; i++){
        shuffle(indices.begin(), indices.end(), rng);
        double epoch_loss = 0.0;
        double epoch_mse = 0.0;
        double epoch_energy = 0.0;
        cout << "Epoch " << i+1 << " starting" << endl;

        get_hyperparameters(i, alpha, lambda);
        cout << "alpha: " << alpha << scientific << setprecision(8) << "; lambda" << lambda << scientific << setprecision(8) << endl;

        for(int j=0;j<num_batches; j++){
            MatrixXd I(6, BATCH_SIZE), T(6, BATCH_SIZE);
            MatrixXd dW1, dW2, dW3; dW1.resizeLike(W_1); dW2.resizeLike(W_2); dW3.resizeLike(W_3);
            VectorXd dB1, dB2, dB3; dB1.resizeLike(B_1); dB2.resizeLike(B_2); dB3.resizeLike(B_3);
            double batch_loss = 0.0, batch_mse = 0.0, batch_energy = 0.0;

            get_batches(data, indices, j, I, T);
            compute_physics_gradients(I, T, W_1, B_1, W_2, B_2, W_3, B_3, dW1, dB1, dW2, dB2, dW3, dB3, batch_loss, batch_mse, batch_energy);
            update_weights(W_1, W_2, W_3, B_1, B_2, B_3, dW1, dW2, dW3, dB1, dB2, dB3);
            epoch_loss += batch_loss;
            epoch_mse += batch_mse;
            epoch_energy += batch_energy;
        }
        cout << "\tTotal Loss: " << scientific << setprecision(8) << epoch_loss << ", MSE Loss: " << epoch_mse << ", Energy Loss: " << epoch_energy << endl;
        cout << defaultfloat; // Reset to default float formatting for subsequent outputs if any
        vector<double> losses = {epoch_loss, epoch_mse, epoch_energy};
        loss_over_time.push_back(losses);
    }

    // Saving loss to file
    save_error_to_file(ERRORS_FILE, loss_over_time);

    OrbitState prev = string_to_orbitstate("2449718.500000000,-2.595232416743753E+07,1.447959489186955E+08,1.015388480179012E+03,-2.982057650057118E+01,-5.367390412066213E+00,-8.642933998290747E-04");
    OrbitState current = string_to_orbitstate("2449719.500000000,-2.852462860978506E+07,1.443095614338196E+08,9.507541560083628E+02,-2.972182852535466E+01,-5.891254968619044E+00,-6.259023919934492E-04");
    OrbitState next = string_to_orbitstate("2449720.500000000,-3.108796792065821E+07,1.437780016267328E+08,9.079658266603947E+02,-2.961307532000635E+01,-6.412983621631280E+00,-3.620638713424107E-04");

    current.ax = (next.vx - prev.vx) / 2.0;
    current.ay = (next.vy - prev.vy) / 2.0;
    current.az = (next.vz - prev.vz) / 2.0;

    simulate_data(SIMULATION_FILE, current, 1000, W_1, W_2, W_3, B_1, B_2, B_3, means, std_dev);
    

    return 0;
}