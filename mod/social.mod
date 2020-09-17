/*********************************************
 * OPL 12.9.0.0 Model
 * Author: yangk
 * Creation Date: Jun 10, 2020 at 1:37:56 PM
 *********************************************/
tuple pickupCostTuple{
	int s;
	int o;
	int tt;
	float ll;
	float c;
}

tuple rebTuple{
	int d;
	int s;
	int tt;
	float ll;
	float prob;
}

tuple demandTuple{
	int o;
	int d;
	int t;
	float v;
}

tuple demandAttrTuple{
	int o;
	int d;
	float f;
	int tt;
	float ll;
	float fn;
}

tuple supplyTuple{
	int o;
	string c;
	int t;
	float v;  
}

tuple edge{
	int i;
	int j;
}

tuple dedge{
	int i;
	int j;
	int t;
}

{string} class = {"auto","non-auto"};
float N = ...;
float beta = ...;
float commission[class] = ...;
float sigma = ...;
float xi = ...;
float vot = ...;
string day = ...;
{rebTuple} reb = ...;
{pickupCostTuple} PickupCost = ...;


string path = ...;
int t0 = ...;
int te = ...;
{demandTuple} Demand = ...;
{demandAttrTuple} demandAttr = ...;
{supplyTuple} accInitTuple = ...;
//{dw_tuple} dwTuple = ...;
{supplyTuple} daccTuple = ...;

{edge} pickupEdge = {<s,o>|<s,o,tt,ll,c> in PickupCost};
int pcktt[pickupEdge] = [<s,o>:tt|<s,o,tt,ll,c> in PickupCost];
float pckll[pickupEdge] = [<s,o>:ll|<s,o,tt,ll,c> in PickupCost];
float pckCost[pickupEdge] = [<s,o>:c|<s,o,tt,ll,c> in PickupCost];

{edge} rebEdge = {<d,s>|<d,s,tt,ll,prob> in reb};
float rebll[rebEdge] = [<d,s>:ll|<d,s,tt,ll,prob> in reb];
int rebtt[rebEdge] = [<d,s>:tt|<d,s,tt,ll,prob> in reb];

float rebProb[rebEdge] = [<s,o>:prob|<s,o,tt,ll,prob> in reb];

{dedge} demandODPair  = {<o,d,t>|<o,d,t,v> in Demand};
{edge} demandEdge  = {<o,d>|<o,d,f,tt,ll,fn> in demandAttr};
float demand[demandODPair] = [<o,d,t>:v|<o,d,t,v> in Demand];
float fare[demandEdge] = [<o,d>:f|<o,d,f,tt,ll,fn> in demandAttr];
int paxtt[demandEdge] = [<o,d>:tt|<o,d,f,tt,ll,fn> in demandAttr];
float paxll[demandEdge] = [<o,d>:ll|<o,d,f,tt,ll,fn> in demandAttr];

float fare_n[demandEdge] = [<o,d>:fn|<o,d,f,tt,ll,fn> in demandAttr];

{int} Region  = {s|<s,c,t,v> in accInitTuple};
float accInit[Region][class] = [r:[c:v]|<r,c,t,v> in accInitTuple];
float dacc[Region][t0..te-1][class] = [r:[t:[c:v]]|<r,c,t,v> in daccTuple];

dvar float+ rebFlow[rebEdge][t0..te-1];
dvar float+ acc[Region][t0..te][class];
dvar float+ demandFlow[demandODPair][class]; 
dvar float+ pickupFlow[pickupEdge][t0..te-1][class]; 
dvar float+ obj;
maximize sum(e in demandODPair, c in class) demandFlow[e][c] * (fare[<e.i,e.j>] - sigma*paxll[<e.i,e.j>])
 - sum(e in pickupEdge, t in t0..te-1, c in class) pickupFlow[e][t][c] * (vot*pcktt[e]+sigma*pckll[e])
  - sum(e in rebEdge, t in t0..te-1) (rebFlow[e][t]+acc[e.i][t]["non-auto"]*rebProb[e]) * rebll[e] * sigma
 + xi * (sum(e in demandODPair) demandFlow[e]["non-auto"] * fare_n[<e.i,e.j>]
 - sum(e in pickupEdge, t in t0..te-1) pickupFlow[e][t]["non-auto"] * pckCost[e]);

//maximize sum(e in demandODPair, c in class) demandFlow[e][c] * fare[<e.i,e.j>] *commission[c]
// - sum(e in demandODPair)sigma*paxll[<e.i,e.j>] *demandFlow[e]["auto"] - sum(e in rebEdge, t in t0..te-1)rebFlow[e][t]* rebll[e] * sigma
//- sum(e in pickupEdge, t in t0..te-1) pickupFlow[e][t]["auto"] * sigma*pckll[e]
// + xi * (sum(e in demandODPair) demandFlow[e]["non-auto"] * fare_n[<e.i,e.j>]
// - sum(e in pickupEdge, t in t0..te-1) pickupFlow[e][t]["non-auto"] * pckCost[e]);
 
subject to{

	obj == sum(e in demandODPair, c in class) demandFlow[e][c] * (fare[<e.i,e.j>] - sigma*paxll[<e.i,e.j>])
 - sum(e in pickupEdge, t in t0..te-1, c in class) pickupFlow[e][t][c] * (vot*pcktt[e]+sigma*pckll[e])
  - sum(e in rebEdge, t in t0..te-1) (rebFlow[e][t]+acc[e.i][t]["non-auto"]*rebProb[e]) * rebll[e] * sigma
 + xi * (sum(e in demandODPair) demandFlow[e]["non-auto"] * fare_n[<e.i,e.j>]
 - sum(e in pickupEdge, t in t0..te-1) pickupFlow[e][t]["non-auto"] * pckCost[e]);
    forall(t in t0..te-1)
	{
		forall(r in Region)
		{			
			acc[r][t+1]["non-auto"] == acc[r][t]["non-auto"] * (1-sum(e in rebEdge:e.i==r)rebProb[e]) 
				+ sum(e in rebEdge:e.j==r) acc[e.i][t]["non-auto"] * rebProb[e] 
				- sum(e in pickupEdge:e.i==r) pickupFlow[e][t]["non-auto"]+ dacc[r][t]["non-auto"];
				
			
			acc[r][t+1]["auto"] == acc[r][t]["auto"] - sum(e in rebEdge:e.i==r ) rebFlow[e][t] + sum(e in rebEdge:e.j==r && t-rebtt[e]+1>=t0) rebFlow[e][t-rebtt[e]+1]
				- sum(e in pickupEdge:e.i==r) pickupFlow[e][t]["auto"]+ dacc[r][t]["auto"];
			
			forall(c in class)
				sum(e in demandODPair:e.i==r && e.t==t) demandFlow[e][c] == sum(e in pickupEdge:e.j==r) pickupFlow[e][t][c];
				
		}		
		
	}
	forall(e in demandODPair)
		  	sum(c in class) demandFlow[e][c] <= demand[e];
			
	forall(r in Region)
	{
		forall(c in class)
			acc[r][t0][c] == accInit[r][c];
   }					
   
  
}
main{
	var file = new IloOplOutputFile(thisOplModel.path);
	t = thisOplModel.t0;
	thisOplModel.generate();
	cplex.solve();
	file.write("rebFlow=[");
	for(var e in thisOplModel.rebEdge)
	{
		if(thisOplModel.rebFlow[e][t]>0){	
	
		file.write("(");
		file.write(e);
		file.write(",");
		file.write(thisOplModel.rebFlow[e][t]);
		file.write(")");	
		}					
	}
	file.writeln("];");
	
	file.write("pickupFlow=[");
	for(var e in thisOplModel.pickupEdge)
	{	  
	 	if(thisOplModel.pickupFlow[e][t]["auto"] >0)
		{
			file.write("(");
			file.write(e);
			file.write(",");
			file.write(thisOplModel.pickupFlow[e][t]["auto"]);
			file.write(")");		
		}	
	}
	file.writeln("];");
	
	file.write("demandFlow=[");
	for(var e in thisOplModel.demandODPair)
	{	 
		if(e.t==t)
		 	if(thisOplModel.demandFlow[e]["auto"] >0)
			{
				file.write("(");
				file.write(e);
				file.write(",");
				file.write(thisOplModel.demandFlow[e]["auto"]);
				file.write(")");		
			}	
	}
	file.writeln("];");
	
	
}