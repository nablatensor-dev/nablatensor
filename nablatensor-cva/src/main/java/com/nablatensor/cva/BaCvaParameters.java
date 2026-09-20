package com.nablatensor.cva;
import com.nablatensor.codegen.Of;
@Of public final class BaCvaParameters {
  private final double rho, beta, alpha;
  private BaCvaParameters(double rho,double beta,double alpha){this.rho=rho;this.beta=beta;this.alpha=alpha;}
  static BaCvaParameters create(double rho,double beta,double alpha){return new BaCvaParameters(rho,beta,alpha);}
  public static BaCvaParametersBuilder of(){return new BaCvaParametersBuilder();}
  public static BaCvaParameters standard(){return of().rho(.5).beta(.25).alpha(1.4).build();}
  public double rho(){return rho;} public double beta(){return beta;} public double alpha(){return alpha;}
  public double riskWeight(CreditName.Rating rating,CreditName.Sector sector){boolean ig=switch(rating){case AAA,AA,A,BBB->true;default->false;};double base=switch(sector){case SOVEREIGN->ig?.005:.020;case LOCAL_GOVERNMENT->ig?.010:.040;case FINANCIAL->ig?.050:.120;case CORPORATE,TECH->ig?.030:.080;case CONSUMER->ig?.030:.070;case OTHER->ig?.035:.120;};return switch(rating){case AAA->base*.7;case AA->base*.85;case A->base;case BBB->base*1.15;case BB->base;case B->base*1.5;case CCC->base*3;case UNRATED->base*1.25;};}
  public double supervisoryDiscount(double maturity){double x=.05*maturity;return x<=0?1:(1-Math.exp(-x))/x;}
}
