#include<iostream>
#include<cstdlib>
#include<cmath>
#include<fstream>
#include<random>
#include<complex>
#include<sstream>

using namespace std;

#define CD const double
#define CI const int
#define CB const bool 

//Computes a sigmoidal function
double sigmoid(CD inpt, CD thr, CD k, CD expthr, CD gain)
{
    if (inpt >= thr)
    {
        double expinpt = exp(-k*(inpt-thr));
        return gain * (1.0 - expinpt) / (1.0 + expthr*expinpt);
    }
    else return 0.0;
}

//Sigmoidal for closing and opening gates
double sigmo_rates(CD inpt, CD k, CD thres, CD gain)
{
    return gain / (1.0 + exp(-k*(inpt - thres)));
}

//Integration step
void step(CI ncls, const vector<vector<int>> &neighs, 
vector<double> &x, vector<double> &r, vector<vector<bool>> &g, 
CD w0, CD x0, CD rc, CD ro, CB nogates, CD b, CD s, const vector<double> &h, CD rmax, CD tc, CD td, 
CD k, CD thr, CD expthr, CD gain, CD k_gate, CD thr_gate,
CD dt, CD sqdt, ofstream &output,
mt19937 &gen, uniform_real_distribution<double> &ran_u, normal_distribution<double> &ran_g)
{
    int i,j ; //Counter

    int neigh, where_am_I; //Indices 

    //Auxiliary variables
    double noise;
    double inpt;
    double prob;



    //Do not overwrite old variables!
    vector<vector<bool>> gates(ncls, vector<bool>(4, true));
    vector<double> xold, rold;

    xold = vector<double>(ncls);
    rold = vector<double>(ncls);

    xold.swap(x);
    rold.swap(r);
    gates.swap(g);

    //Update clusters
    for (i=0; i < ncls; i++)
    {
        inpt = 0.0;
        //Update gates of this cluster
        for (j=0; j < neighs[i].size(); j++)
        {
            neigh = neighs[i][j];

            //Localize my index in the array of the neighbour
            where_am_I = 0;
            while (neighs[neigh][where_am_I] != i)
            {
                where_am_I++;
            } 

            //Sum input TO this module
            inpt += w0 * xold[neigh] * (nogates || gates[neigh][where_am_I]);  //gate true = information can pass


            //Update now gates FROM this module
            if (gates[i][j])
            {
                prob = 1.0 - exp(-dt * (1.0 - sigmo_rates(rold[i], k_gate, thr_gate, rc)));
                g[i][j] = ran_u(gen) > prob; //Make false with probability prob
            }
            else
            {
                prob = ro;
                g[i][j] = ran_u(gen) < prob; //Make true with prob
            }
            
        }

        //Noise
        noise = s * ran_g(gen);
        noise += xold[i] > x0 ? 2*s*sqrt(xold[i])*ran_g(gen) : 0.0;

        //Milstein algorithm for Ito equations
        x[i] = xold[i] + dt * (-b*(xold[i]-x0) + sigmoid(rold[i] * (xold[i] + inpt + h[i]), thr, k, expthr, gain) ) + sqdt * noise; 
        r[i] = rold[i] + dt * ((rmax - rold[i])/tc - rold[i]*xold[i]/td);
    }
}

//Create a lattice
void create_system(const int L, const int ncls, vector<vector<int>> &neighs)
{
    int x,y,j;
    int last;

    neighs = vector<vector<int>>(ncls);

    //Corners...
    neighs[0].push_back(1); //Low left
    neighs[0].push_back(L);

    neighs[L-1].push_back(L-2); //Low right
    neighs[L-1].push_back(2*L-1);

    neighs[L*(L-1)].push_back(L*(L-1)+1); //Up left
    neighs[L*(L-1)].push_back(L*(L-1)-L);

    neighs[(L+1)*(L-1)].push_back((L+1)*(L-1)-1); //Up right
    neighs[(L+1)*(L-1)].push_back((L+1)*(L-1)-L);


    //Small system only has corners
    if (L==2) return;

    //First row/col
    //x=0, y=0, x=L-1 and y=L-1
    last = L-1;
    for (j=1; j < L-1; j++)
    {
        //y=0
        neighs[j].push_back(j+1);
        neighs[j].push_back(j-1);
        neighs[j].push_back(L+j);

        //y=L-1
        neighs[L*last + j].push_back(j+1);
        neighs[L*last + j].push_back(j-1);
        neighs[L*last + j].push_back(L+j);

        //y=0
        neighs[L*j].push_back(L*(j+1));
        neighs[L*j].push_back(L*(j-1));
        neighs[L*j].push_back(L*j+1);

        //y=L-1
        neighs[L*j + L-1].push_back(L*(j+1));
        neighs[L*j + L-1].push_back(L*(j-1));
        neighs[L*j + L-1].push_back(L*j-1);
    }


    //System inside
    for (y=1; y < L-1; y++)
    {
        for (x=1; x<L-1; x++)
        {
            last = x+y*L;
            neighs[last].push_back(last+1);
            neighs[last].push_back(last-1);
            neighs[last].push_back(last+L);
            neighs[last].push_back(last-L);
        }
    }

    for (x=0; x < neighs.size(); x++)
    {
        cout << x << " : ";
        for (y=0; y < neighs[x].size(); y++)
        {
            cout << neighs[x][y] << " ";
        }
        cout << endl;
    }

    return;
}

//TODO make the gating model simulation here
int main(int argc, char *argv[])
{
    //Counters
    int i,j;

    //System size
    int ncls, L;

    //Time integration
    double t, dt, sqdt, tf, trelax;

    //Gate variables
    double rate_close, rate_open;
    bool nogates;

    //Module activity vars
    double beta, sigma, w0, x0, h;
    vector<double> h_ext;

    //Module recovery vars
    double rmax;
    double tau_c, tau_d;

    //Sigmoidal related functions
    double k, thr, gain, expthr;
    double k_gate, thr_gate;

    //Initial conditions
    int nbins, nbins_relax;

    //Variables
    vector<double> x;
    vector<double> r;
    vector<vector<int>> neighs;
    vector<vector<bool>> gate;

    //RNG
    unsigned int rseed;
    mt19937 gen;
    uniform_real_distribution<double> ran_u(0.0,1.0);
    normal_distribution<double> ran_g(0.0,1.0);

    //Output stream
    string filepath;
    ofstream output;

    //Grab parameters
    if (argc == 14)
    {
        rate_close  = stod(argv[1]);
        rate_open   = stod(argv[2]);
        beta        = stod(argv[3]);
        sigma       = stod(argv[4]);
        h           = stod(argv[5]);
        rmax        = stod(argv[6]);
        tau_c       = stod(argv[7]);
        tau_d       = stod(argv[8]);
        tf          = stod(argv[9]);
        trelax      = stod(argv[10]);
        dt          = stod(argv[11]); 
        L           = stoi(argv[12]);
        filepath    = argv[13];
    }
    else if (argc == 9)
    {
        rate_close  = 1.0; 
        rate_open   = 0.1;
        beta        = 1.55;
        sigma       = 0.1;
        h           = stod(argv[1]); //usual stimulation: 1.5
        w0          = stod(argv[2]);
        x0          = 0.01;
        rmax        = 2.0; 
        tau_c       = 5.0;
        tau_d       = 0.55; 
        tf          = stod(argv[3]);
        trelax      = stod(argv[4]);
        dt          = 0.01;
        L           = stoi(argv[5]);
        nogates     = stoi(argv[6]);
        rseed       = stoi(argv[7]);
        filepath    = argv[8];
    }
    else 
    {
        cout << "ERROR" << argc << endl;
        return EXIT_SUCCESS;
    }

    //Init RNG 
    gen.seed(rseed);

    //Crete system
    ncls = L*L;
    x = vector<double>(ncls);
    r = vector<double>(ncls);
    h_ext = vector<double>(ncls, h);
    gate = vector<vector<bool>>(ncls, vector<bool>(4, true));
    create_system(L, ncls, neighs);

    //Get some variables that we need 
    nbins = int(tf/dt);
    nbins_relax = int(trelax/dt);
    sqdt = sqrt(dt);

    //Sigmoidal variables
    k = 1.6;
    thr = 0.4;
    expthr = exp(k*thr);
    gain = 10.0;


    //Gate variables
    rate_open  = 1.0 - exp(-rate_open * dt);

    k_gate = 30.0;
    thr_gate = 0.55;

    //Initial conditions
    for (i=0; i < ncls; i++)
    {
        x[i] = ran_u(gen);
        r[i] = ran_u(gen) * rmax;
    } 

    //Perform simulation
    output.open(filepath + ".csv");

    //CSV file header
    output << ",time,";
    for (j=1;j<=ncls-1;j++) output << "mod_"<< j << ",mod_" << j <<"_res,";
    output << "mod_" << ncls << ",mod_" << ncls << "_res" <<endl;

    for (j=0; j < nbins; j++)
    {
        step(ncls, neighs, x, r, gate, w0, x0, rate_close, rate_open, nogates, beta, sigma, h_ext, rmax, tau_c, tau_d, k, thr, expthr, gain, k_gate, thr_gate, dt, sqdt, output, gen, ran_u, ran_g);
        t = j*dt;
        //if (j%10 == 0)
        //{
        output << j << "," << t << ",";
        for (i=0; i < ncls-1; i++) output << x[i] << "," << r[i] << ",";
        output  << x[ncls-1] << "," << r[i] << endl; 
        //}
    }
    output.close();

    return EXIT_SUCCESS;
}