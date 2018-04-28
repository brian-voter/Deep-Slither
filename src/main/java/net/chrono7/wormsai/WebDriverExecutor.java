//C:\Users\Brian\IdeaProjects\WormsAI\store\extensions

package net.chrono7.wormsai;

import net.lightbody.bmp.BrowserMobProxy;
import net.lightbody.bmp.BrowserMobProxyServer;
import org.openqa.selenium.*;
import org.openqa.selenium.Dimension;
import org.openqa.selenium.Point;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.chrome.ChromeOptions;
import org.openqa.selenium.remote.CapabilityType;

import java.awt.*;
import java.awt.Rectangle;
import java.awt.event.InputEvent;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;

import static org.apache.commons.io.FileUtils.writeStringToFile;

public class WebDriverExecutor {

    private static final int PIXELS_RIGHT = 20;
    private static final int PIXELS_LEFT = 30;
    private static final int PIXELS_DOWN = 110;
    private static final int PIXELS_UP = 120;
    private static final Dimension WINDOW_SIZE = new Dimension(1920, 1080);
    private ChromeDriver driver;
    private WebElement game;
    private Robot robot;
    private int lastMouseX, lastMouseY;

    WebDriverExecutor() {

        BrowserMobProxy proxy = new BrowserMobProxyServer();
        proxy.start(18904);
        proxy.blacklistRequests("http://slither.io/s/bg54.jpg", 204);
        proxy.blacklistRequests("http://slither.io/s/gbg.jpg", 204);
        //Finish setting up your driver

        try {
            Runtime.getRuntime().exec("taskkill /F /IM chromedriver.exe");
        } catch (IOException e) {
            e.printStackTrace();
        }

        Runtime.getRuntime().addShutdownHook(new Thread() {
            @Override
            public void run() {
                proxy.stop();
                driver.close();
//                service.stop();
                System.exit(0);
            }
        });

        System.setProperty("webdriver.chrome.driver", "C:\\Users\\Brian\\IdeaProjects\\WormsAI\\store\\drivers\\chromedriver.exe");

        // Get BrowserMobProxy server port for PAC file
        int proxyPort = proxy.getPort();

        // Create PAC file to send WebSocket requests direct but other protocols throught BrowserMobProxy server
        String pacFunction = "function FindProxyForURL(url, host) { if (url.substring(0, 3) === \"ws:\" || url.substring(0, 4) === \"wss:\") { " +
                "return \"DIRECT\"; } else { return \"PROXY 127.0.0.1:" + proxyPort + "\"; } }";
        File pacFile = new File("C:/Users/Brian/IdeaProjects/WormsAI/store/misc/proxy.pac");
        try {
            writeStringToFile(pacFile, pacFunction, "UTF-8");
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Set PAC file path to Selenium proxy
        Proxy seleniumProxy = new Proxy();
        seleniumProxy.setProxyType(Proxy.ProxyType.PAC);
        seleniumProxy.setProxyAutoconfigUrl(String.valueOf(pacFile.toURI()));

        ChromeOptions options = new ChromeOptions();
        options.setExperimentalOption("useAutomationExtension", false);
        options.setExperimentalOption("excludeSwitches",
                Collections.singletonList("enable-automation"));
        options.setCapability(CapabilityType.PROXY, seleniumProxy);

        driver = new ChromeDriver(options);

        try {
            robot = new Robot();
        } catch (AWTException e) {
            e.printStackTrace();
        }
    }

    public void quitDriver() {
        driver.quit();
    }


    private void delay(long period) {
        try {
            Thread.sleep(period);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    public void navigate() {
        driver.manage().window().setSize(WINDOW_SIZE);
        driver.get("http://slither.io");
        WebElement e = driver.findElement(By.id("nick"));

        delay(2000);

//        e.sendKeys("TEST-301");

//        delay(1000);

        driver.findElement(By.id("grqi")).click();

        WebElement play = driver.findElements(By.className("nsi")).stream()
                .filter(e1 -> e1.getText().equals(" Play ")).findFirst().orElseGet(null);

        if (play != null) {
            play.click();
        }
        delay(5000);

        game = driver.findElement(By.cssSelector("body"));


        //todo: don't filter out the game (canvas)?

        String[] elementsToHide = new String[] {"/html/body/div[9]", "/html/body/div[10]",
                "/html/body/div[11]", "/html/body/div[12]", "/html/body/div[13]",
                "/html/body/div[15]"};

        Arrays.stream(elementsToHide).map(driver::findElementByXPath).forEach(this::hideElement);

    }

//    public void point(int x, int y) {
//        Point tL = getTopLeftPoint();
//
//        lastMouseX = tL.getX() + x;
//        lastMouseY = tL.getY() + y;
//        robot.mouseMove(lastMouseX, lastMouseY);
//    }

    public void setBoost(boolean boost) {
        if (boost) {
            robot.mousePress(InputEvent.BUTTON1_DOWN_MASK);
        }else {
            robot.mouseRelease(InputEvent.BUTTON1_DOWN_MASK);
        }
    }

    private int attemptScoreRetrieval() throws StaleElementReferenceException{
        return Integer.valueOf(driver.findElementByXPath("/html[1]/body[1]/div[13]/span[1]/span[2]")
                .getAttribute("textContent"));
    }

    public int getScore() {
        int score = 0;
        boolean retrieved = false;
        int attempts = 0;

        while (!retrieved) {
            try{
                score = attemptScoreRetrieval();
                retrieved = true;
            } catch (StaleElementReferenceException e){
                attempts++;

                if (attempts > 10){
                    throw new StaleElementReferenceException("Failed to retrieve score after 10 attempts.");
                }
            }
        }

        return score;
    }

    private void hideElement(WebElement element) {
        driver.executeScript("arguments[0].setAttribute('style', arguments[0].getAttribute('style') + 'visibility:hidden;');",
                element);
    }

//    public void setAttribute(WebElement element, String attName, String attValue) {
//        driver.executeScript("arguments[0].setAttribute(arguments[1], arguments[2]);",
//                element, attName, attValue);
//    }

//    public BufferedImage getScreenshot() {
//        try {
//            return ImageIO.read(new ByteArrayInputStream(game.getScreenshotAs(OutputType.BYTES)));
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
//
//        return null;
//    }

    public BufferedImage getScreenshot() {

        Point tl = getTopLeftPoint();

        BufferedImage image = robot.createScreenCapture(new Rectangle(tl.getX(),
                tl.getY(), WINDOW_SIZE.width - PIXELS_LEFT, WINDOW_SIZE.height - PIXELS_UP));

//        BufferedImage image = robot.createScreenCapture(new Rectangle(game.getLocation().getX(),
//                game.getLocation().getY(), game.getSize().getWidth(), game.getSize().getHeight()));

//        try {
//            ImageIO.write(image, "PNG", new File("out.png"));
//        } catch (IOException e) {
//            e.printStackTrace();
//        }

        return image;
    }

    public Point getTopLeftPoint() {
        return new Point(game.getLocation().getX() + PIXELS_RIGHT, game.getLocation().getY() + PIXELS_DOWN);
    }

    public Point getCenterPoint() {
        return new Point((game.getLocation().getX() + PIXELS_RIGHT + game.getSize().width - PIXELS_LEFT) / 2,
                (game.getLocation().getY() + PIXELS_DOWN + (game.getSize().height - PIXELS_UP) / 2));
    }

    public java.awt.Point getMousePoint() {
        return new java.awt.Point(lastMouseX, lastMouseY);
    }

    public void point(java.awt.Point point) {
        robot.mouseMove((int) point.getX(), (int) point.getY());
    }

    public void pointAdjusted(java.awt.Point point) {
        Point tl = getTopLeftPoint();
        robot.mouseMove((int) point.getX() + tl.x, (int) point.getY() + tl.y);
    }

    public void fixLoss() {
        try {
            WebElement play = driver.findElementByXPath("/html[1]/body[1]/div[2]/div[5]/div[1]/div[1]/div[3]");

            play.click();

            Thread.sleep(1000);

            WebElement ad = driver.findElementByXPath("/html[1]/body[1]/div[17]");
            ad.click();

        } catch (Exception e) {

        }
    }
}