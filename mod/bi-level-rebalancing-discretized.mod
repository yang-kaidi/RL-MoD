/*********************************************
 * OPL 12.10.0.0 Model
 * Author: yangk
 * Creation Date: Aug 25, 2020 at 10:45:21 AM
 *********************************************/
tuple demandAttrTuple{
  	int i;
  	int j;
  	int t;
  	float v;
  	float p;
}

tuple edgeAttrTuple{
  	int i;
  	int j;
  	int t;
}

tuple Edge{
  int i;
  int j;
}

tuple demandEdgeTuple{
  int i;
  int j;
  int t;
}

tuple accTuple{
  int i;
  float n;
}

tuple daccTuple{
  int i;
  int t;
  float n;
}

string path = ...;
int t0 = ...;
int T = ...;
int tf = t0+T;
float beta = ...;
{demandAttrTuple} demandAttr = ...;
{edgeAttrTuple} edgeAttr = ...;
{accTuple} accInitTuple = ...;
{daccTuple} daccAttr = ...;
{float} actions = ...;
{Edge} edge = {<i,j>|<i,j,t> in edgeAttr};
{int} region = {i|<i,v> in accInitTuple};
float accInit[region] = [i:v|<i,v> in accInitTuple];
float dacc[region][t0..tf-1] = [i:[t:v]|<i,t,v> in daccAttr];
{demandEdgeTuple} demandEdge = {<i,j,t>|<i,j,t,v,p> in demandAttr};
float demand[demandEdge] = [<i,j,t>:v|<i,j,t,v,p> in demandAttr];
float price[demandEdge] = [<i,j,t>:p|<i,j,t,v,p> in demandAttr];
int tt[edge] = [<i,j>:t|<i,j,t> in edgeAttr];
dvar float+ demandFlow[edge][t0..tf-1];
dvar float+ rebFlow[edge][t0..tf-1];
dvar float+ acc[region][t0..tf];
dvar float+ rho[region][t0..tf-1];
dvar float+ pi[region][t0..tf-1];
dvar float+ xi[edge][t0..tf-1];
dvar int+ desiredAcc[region][t0..tf];
dvar boolean u[region][t0..tf-1];
dvar boolean v[region][t0..tf-1];
dvar boolean w[edge][t0..tf-1];
dvar boolean action[region][actions][t0..tf-1];


maximize(sum(e in demandEdge) demandFlow[<e.i,e.j>][e.t]*price[e] - beta * sum(e in edge,t in t0..tf-1)(rebFlow[e][t]+demandFlow[e][t])*tt[e]);
subject to
{
  forall(t in t0..tf-1)
  {
    forall(i in region)
    {  
    	acc[i][t+1] == acc[i][t] - sum(e in edge: e.i==i)(demandFlow[e][t] + rebFlow[e][t]) 
      			+ sum(e in edge: e.j==i && t-tt[e]>=t0)(demandFlow[e][t-tt[e]] + rebFlow[e][t-tt[e]]) + dacc[i][t];
		sum(e in edge: e.i==i)(demandFlow[e][t]+ rebFlow[e][t]) <= acc[i][t];
		sum(e in edge: e.i==i && e.i!=e.j) (rebFlow[<e.j, e.i>][t] - rebFlow[<e.i, e.j>][t]) - desiredAcc[i][t] + acc[i][t] - sum(e in edge: e.i==i)demandFlow[e][t] >=0;
		sum(e in edge: e.i==i && e.i!=e.j) (rebFlow[<e.j, e.i>][t] - rebFlow[<e.i, e.j>][t]) - desiredAcc[i][t] + acc[i][t] - sum(e in edge: e.i==i)demandFlow[e][t] <= 1000 * u[i][t];
		forall(act in actions)
		{
			desiredAcc[i][t] >= (sum(ii in region) acc[ii][t] - sum(e in edge)demandFlow[e][t])*act -1 + 1e-3 - 40 * (1-action[i][act][t]);
			desiredAcc[i][t] <= (sum(ii in region) acc[ii][t] - sum(e in edge)demandFlow[e][t])*act + 40 * (1-action[i][act][t]);
		}
		sum(act in actions) action[i][act][t] == 1;
		acc[i][t] - sum(e in edge: e.i==i)rebFlow[e][t]  - sum(e in edge: e.i==i)demandFlow[e][t]<= 1000 * v[i][t];
		rho[i][t] <= 1000 * (1-u[i][t]);
		pi[i][t] <= 1000 * (1-v[i][t]);
      	if(t == t0)
      		acc[i][t] == accInit[i];
 	}  	    
    forall(e in edge)
    {
		if(<e.i,e.j,t> in demandEdge)
      		demandFlow[e][t] <= demand[<e.i,e.j,t>];
		else
      		demandFlow[e][t] == 0;      
		tt[e] - rho[e.j][t] + rho[e.i][t] + pi[e.i][t] - xi[e][t] == 0;
		rebFlow[e][t] <= 1000 * (1-w[e][t]);
		xi[e][t] <= 1000 * w[e][t];
		
	}
  }
  
}

main {
  thisOplModel.generate();
  cplex.solve();
  var t = thisOplModel.t0
  var ofile = new IloOplOutputFile(thisOplModel.path);
  ofile.write("flow=[")
  for(var e in thisOplModel.edge)
	if(thisOplModel.demandFlow[e][t]>1e-3 || thisOplModel.rebFlow[e][t]>1e-3)
       {
         ofile.write("(");
         ofile.write(e.i);
         ofile.write(",");
         ofile.write(e.j);
         ofile.write(",");
         ofile.write(thisOplModel.demandFlow[e][t]);
		 ofile.write(",");
		 ofile.write(thisOplModel.rebFlow[e][t]);
         ofile.write(")");
       }
  ofile.writeln("];")
  ofile.write('desiredAcc=[')
  for(var i in thisOplModel.region)
  {
	 ofile.write("(");
	 ofile.write(i);
	 ofile.write(",");
	 ofile.write(thisOplModel.desiredAcc[i][t]);
	 ofile.write(")");
  }
  ofile.writeln("];")
  ofile.close();
}




