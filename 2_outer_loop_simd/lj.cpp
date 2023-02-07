#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cmath>

#include <string>

#include "profiler.hpp"

#include <ittnotify.h>

//#define DEBUG

struct Vec3{
  double x,y,z;
  Vec3() {}
  Vec3(const double s) : x(s),y(s),z(s) {}
  Vec3(const double _x,const double _y,const double _z) : x(_x), y(_y), z(_z) {}

  Vec3 operator+(const Vec3& rhs)   const { return Vec3(this->x + rhs.x,this->y + rhs.y,this->z + rhs.z); }
  Vec3 operator-(const Vec3& rhs)   const { return Vec3(this->x - rhs.x,this->y - rhs.y,this->z - rhs.z); }
  Vec3 operator*(const Vec3& rhs)   const { return Vec3(this->x * rhs.x,this->y * rhs.y,this->z * rhs.z); }
  Vec3 operator*(const double& rhs) const { return Vec3(this->x * rhs,  this->y * rhs,  this->z * rhs); }
  Vec3 operator/(const double& rhs) const { return Vec3(this->x / rhs,  this->y / rhs,  this->z / rhs); }
  void operator+=(const Vec3& rhs)   { *this = *this + rhs; }
  void operator-=(const Vec3& rhs)   { *this = *this - rhs; }
  void operator*=(const Vec3& rhs)   { *this = *this * rhs; }
  void operator*=(const double& rhs) { *this = *this * rhs; }
  void operator/=(const double& rhs) { *this = *this / rhs; }
  friend double sum(const Vec3& v) { return v.x + v.y + v.z; }
  friend Vec3 operator*(const double& lhs, const Vec3& rhs) { return Vec3(rhs.x*lhs, rhs.y*lhs, rhs.z*lhs); }
};

struct Atom{
  Vec3 pos,vel,force;
};

namespace FCC{
  const int nunit = 4;
  double unit_length = 1.0;
};

void generate_fcc_structure(const int nmol, const int nlattice, Atom* atom){
  assert(atom != nullptr);
  // set unit lattice
  const Vec3 unit[4] = {
	  Vec3(0.0,0.0,0.0),
	  Vec3(FCC::unit_length*0.5,FCC::unit_length*0.5,0.0),
	  Vec3(FCC::unit_length*0.5,0.0,FCC::unit_length*0.5),
	  Vec3(0.0,FCC::unit_length*0.5,FCC::unit_length*0.5)};
  //fillout fcc lattice
  int n = 0;
  for(int x = 0; x < nlattice; x++){
  for(int y = 0; y < nlattice; y++){
  for(int z = 0; z < nlattice; z++){
    for(int u = 0; u < FCC::nunit; u++){
      atom[n].pos.x = FCC::unit_length * x + unit[u].x;
      atom[n].pos.y = FCC::unit_length * y + unit[u].y;
      atom[n].pos.z = FCC::unit_length * z + unit[u].z;
#ifdef DEBUG
      fprintf(stdout,"%d %lf %lf %lf\n",n,atom[n].pos.x,atom[n].pos.y,atom[n].pos.z);
#endif
      n++;
    }
  }}}
}

double random_number(){
  return rand() / (double)RAND_MAX;
}
void generate_random_velocity(const int nmol, Atom* atom){
  assert(atom != nullptr);

  for(int i=0;i<nmol;i++){
    atom[i].vel.x = random_number();
    atom[i].vel.y = random_number();
    atom[i].vel.z = random_number();
#ifdef DEBUG
    fprintf(stdout,"%d %lf %lf %lf\n",i,atom[i].vel.x,atom[i].vel.y,atom[i].vel.z);
#endif
  }
}

double calc_kinetic_energy(const int nmol, Atom* atom){
  double ret = 0.0;
  for(int i=0;i<nmol;i++){
    ret += sum(atom[i].vel * atom[i].vel);
  }
  ret *= 0.5;
  return ret;
}

void scale_velocity(const int nmol, const double temperature,Atom* atom){
  const double factor = sqrt(1.5 * nmol * temperature / calc_kinetic_energy(nmol,atom));
  for(int i=0;i<nmol;i++) atom[i].vel *= factor;
}

void remove_system_momentum(const int nmol, Atom* atom){
  Vec3 mom(0.0);
  for(int i=0;i<nmol;i++){
    mom += atom[i].vel;
  }

  mom = mom / (double)nmol;

  for(int i=0;i<nmol;i++){
    atom[i].vel -= mom;
  }
}

void kick(const int nmol, const double dt, Atom* atom){
  for(int i=0;i<nmol;i++){
    atom[i].vel += 0.5 * dt * atom[i].force; // assuming mass = 1.0
  }
}
void drift(const int nmol, const double dt, Atom* atom){
  for(int i=0;i<nmol;i++){
    atom[i].pos += dt * atom[i].vel;
  }
}

void apply_pbc(const int nmol,const double length, Atom* atom){
#ifdef PBC
  for(int i=0;i<nmol;i++){
    if(atom[i].pos.x < 0.0)     atom[i].pos.x += length;
    if(atom[i].pos.x >= length) atom[i].pos.x -= length;
    if(atom[i].pos.y < 0.0)     atom[i].pos.y += length;
    if(atom[i].pos.y >= length) atom[i].pos.y -= length;
    if(atom[i].pos.z < 0.0)     atom[i].pos.z += length;
    if(atom[i].pos.z >= length) atom[i].pos.z -= length;
  }
#endif
}

void calc_force(const int nmol, const double length, Atom* atom){
  const double rcut = (length*0.5 < 3.5) ? length*0.5 : 3.5;
  const double rcut2 = rcut * rcut;
  const double c14 = 48.0;
  const double c8  = 24.0;
  const double lh  = 0.5 * length;
  const double lh_inv = 1.0 / lh;
  const double eps = 1e-32;
#pragma omp parallel for simd
  for(int i=0;i<nmol;i++){
    const Vec3 posi = atom[i].pos;
    Vec3 fi = Vec3(0.0,0.0,0.0);
    for(int j=0;j<nmol;j++){
      const Vec3 posj = atom[j].pos;
      Vec3 dr = posi - posj;
#ifdef PBC
      if(dr.x <= -lh) dr.x += length;
      if(dr.x >   lh) dr.x -= length;
      if(dr.y <= -lh) dr.y += length;
      if(dr.y >   lh) dr.y -= length;
      if(dr.z <= -lh) dr.z += length;
      if(dr.z >   lh) dr.z -= length;
#endif
      const double r2 = eps + dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;
#ifdef CUTOFF
      if(r2 <= rcut2){
#endif
        const double r2_inv = 1.0 / r2;
        const double r6_inv = r2_inv * r2_inv * r2_inv;
        const double f = (c14 * r6_inv - c8) * r6_inv * r2_inv; // assuming sigma = epsilon = 1.0
	fi += f * dr;
#ifdef CUTOFF
      }
#endif
    }
    atom[i].force = fi;
  }
}

double calc_potential_energy(const int nmol, const double length, Atom* atom){
  const double rcut = (length*0.5 < 3.5) ? length*0.5 : 3.5;
  const double rcut2 = rcut * rcut;
  const double lh  = 0.5 * length;
  double pot = 0.0;
  for(int i=0;i<nmol;i++){
    const Vec3 posi = atom[i].pos;
    for(int j=0;j<nmol;j++){
      if(i == j) continue;
      const Vec3 posj = atom[j].pos;
      Vec3 dr = posi - posj;
#ifdef PBC
      if(dr.x <= -lh) dr.x += length;
      if(dr.x >   lh) dr.x -= length;
      if(dr.y <= -lh) dr.y += length;
      if(dr.y >   lh) dr.y -= length;
      if(dr.z <= -lh) dr.z += length;
      if(dr.z >   lh) dr.z -= length;
#endif
      const double r2 = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;
#ifdef CUTOFF
      if(r2 <= rcut2){
#endif
        const double r2_inv = 1.0 / r2;
        const double r6_inv = r2_inv * r2_inv * r2_inv;
        pot += 4.0 * (r6_inv - 1.0) * r6_inv;
#ifdef CUTOFF
      }
#endif
    }
  }
  return 0.5 * pot;
}

int main(int argc, char** argv){
  int nlattice = 4;
  int nmol = FCC::nunit * nlattice * nlattice * nlattice; // number of molecules in fcc structure
  int nstep = 1000;
  int nstep_eq = 1000;
  int interval_energy = 100;
  double density = 1.05;
  double dt = 0.001;
  if(argc > 1){
    int narg = 1;
    while(narg < argc){
      std::string opt = argv[narg++];
      if(opt == "-N" || opt == "--nmol"){
        nlattice = 0;
	nmol = atoi(argv[narg++]);
        int nmol_tmp = FCC::nunit * nlattice * nlattice * nlattice;
        while(nmol_tmp < nmol){
	  nlattice++;
          nmol_tmp = FCC::nunit * nlattice * nlattice * nlattice;
        }
	if(nmol != nmol_tmp){
	  fprintf(stderr,"Warning: Number of molecules is adjusted to fit in FCC structure\n");
	}
        nmol = nmol_tmp;
        fprintf(stderr,"Number of molcules: %d\n",nmol);
      }
      if(opt == "-s" || opt == "--nstep"){
	nstep = atoi(argv[narg++]);
        fprintf(stderr,"Number of steps: %d\n",nstep);
      }
      if(opt == "-e" || opt == "--nstep_eq"){
	nstep_eq = atoi(argv[narg++]);
        fprintf(stderr,"Number of steps for equiribration: %d\n",nstep_eq);
      }
      if(opt == "-d" || opt == "--density"){
	density = atof(argv[narg++]);
        fprintf(stderr,"Density: %lf\n",density);
      }
    }
  }
  FCC::unit_length = powf(FCC::nunit/density,1.0/3.0);
  const double length = nlattice * FCC::unit_length;
  const double dth = 0.5 * dt;
  fprintf(stderr,"System size: [0.0,%lf), [0.0,%lf), [0.0,%lf)\n",length,length,length);

  Atom *atom = nullptr;
  atom = new Atom[nmol];

  generate_fcc_structure(nmol,nlattice,atom);
  generate_random_velocity(nmol,atom);
  remove_system_momentum(nmol,atom);
  scale_velocity(nmol,1.0,atom);

  Profiler pTotal("total"), pForce("force");
  for(int s=-nstep_eq;s<nstep;s++){
    if(s<0) scale_velocity(nmol,1.0,atom);
    if(s==0){
      pTotal.clear();
      pForce.clear();
      pTotal.start();
      __itt_resume();
    }
    kick(nmol,dt,atom);
    drift(nmol,dt,atom);
    apply_pbc(nmol,length,atom);
    pForce.start();
    calc_force(nmol,length,atom);
    pForce.end();
    kick(nmol,dt,atom);
#ifdef TEST
    if(s==0) fprintf(stdout,"#pot kin tot\n");
    if(s >= 0 && s%interval_energy == 0){
      double pot = calc_potential_energy(nmol,length,atom);
      double kin = calc_kinetic_energy(nmol,atom);
      fprintf(stdout,"%4d %lf %lf %lf\n",s,pot,kin,pot+kin);
    }
#endif
  }
  __itt_pause();
  pTotal.end();
  pTotal.print();
  pForce.print();

  if(atom != nullptr) delete[] atom;
}
