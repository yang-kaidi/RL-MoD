tuple pickupCostTuple{
	int o;
	int d;
	float c;
	float t;
}

tuple demandTuple{
	int o;
	int d;
	float v;  
	float f;
}

tuple supplyTuple{
	int o;
	float v;  
}

tuple edge{
	int s;
	int o;
}

tuple sedge{
	int s;
	int o;
	int d;
}
string path = ...;
{pickupCostTuple} PickupCost = ...;
{edge} pickupEdge = {<s,o>|<s,o,c,t> in PickupCost};
{demandTuple} Demand = ...;
{supplyTuple} Supply = ...;
{edge} demandODPair  = {<o,d>|<o,d,v,f> in Demand};
{sedge} solutionEdge = ...;
{int} supplyRegion  = {s|<s,v> in Supply};
float demand[demandODPair] = [<o,d>:v|<o,d,v,f> in Demand];
float fare[demandODPair] = [<o,d>:f|<o,d,v,f> in Demand];
float supply[supplyRegion] = [o:v|<o,v> in Supply];
float cost[pickupEdge] = [<s,o>:c|<s,o,c,t> in PickupCost];

float pickuptt[pickupEdge] = [<s,o>:t|<s,o,c,t> in PickupCost];

dvar float+ flow[solutionEdge];
dvar float+ rebflow[pickupEdge];

//maximize sum(e in solutionEdge: fare[<e.o,e.d>] - cost[<e.s,e.o>]>=0)flow[e]*(fare[<e.o,e.d>]-pickuptt[<e.s,e.o>]*0.1);
maximize sum(e in solutionEdge)flow[e]*(fare[<e.o,e.d>] - cost[<e.s,e.o>]);

subject to{
	forall(od in demandODPair)
		sum(e in solutionEdge: e.o==od.s && e.d == od.o)flow[e] <= demand[od];
	
	forall(n in supplyRegion)
	  	//sum(e in solutionEdge: e.s == n)flow[e] + sum(e in pickupEdge:e.s==n)rebflow[e] == supply[n];
		sum(e in solutionEdge: e.s == n)flow[e] <= supply[n];

}

execute{
	var file = new IloOplOutputFile(thisOplModel.path);	

	for(var e in thisOplModel.solutionEdge)
	{
		if(thisOplModel.flow[e]>0)
		{
			file.write("|");
			file.write(e);
			file.write(",");
			file.write(thisOplModel.flow[e]);
			file.write("|");
		}
	}	
	file.writeln()
}